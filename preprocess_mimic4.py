"""Part 1 - preprocessing: MIMIC-IV -> tokenised ICU trajectories.

Usage:
    python part1_preprocessing/preprocess_mimic4.py --mimic_dir /path/to/mimiciv/3.1 \
        --config part2_configs/model_config.yaml --out data/processed

Pipeline (manuscript Section 2.4):
  1. Cohort: adults, ICU LOS >= 24 h, >= 48 h of charted monitoring; first ICU stay per patient.
  2. Patient-level train/val/test split.
  3. Everything that is *fitted* (vital cut-points, medians, vocabulary, baseline scaling,
     inter-event-time quantiles) is estimated on the TRAINING split only and applied unchanged
     to validation/test (no leakage).
  4. Temporal alignment: vitals -> hourly bins, labs -> daily bins.
  5. Missing data: forward-fill then training-median imputation (continuous), missingness
     indicator kept as auxiliary feature.

ASSUMPTIONS / LIMITATIONS (verify against how the reported results were produced):
  * Item ids below are the MIMIC-IV v3.x ids as we remember them - check them in d_items / d_labitems.
  * MIMIC-IV diagnoses (diagnoses_icd) carry no timestamp, so they enter as baseline multi-hot
    comorbidity features, not as timed events.
  * Vitals are tokenised as low/normal/high (training 10th/90th percentile) and an event is emitted
    only when the category changes. Labs: one token per day (last value) with normal/abnormal flag.
    Medications and procedures: one token per order/procedure start.
  * Sequences longer than max_len-1 events are truncated, then <EOS> is appended.
  * "At least 48 h of continuous monitoring" is implemented as >= 48 h between first and last charted vital.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import yaml

VITALS = {220045: "HR", 220179: "SBP", 220180: "DBP", 220181: "MAP", 220210: "RR", 220277: "SPO2", 223762: "TEMPC"}
LABS = {50912: "CREAT", 50813: "LACTATE", 50971: "K", 50983: "NA", 51222: "HGB", 51301: "WBC",
        51265: "PLT", 50882: "HCO3", 50931: "GLU", 51006: "BUN", 50820: "PH"}
PAD, EOS = 0, 1


def read(mimic, rel, usecols, **kw):
    path = os.path.join(mimic, rel)
    if not os.path.exists(path):
        path = path.replace(".csv.gz", ".csv")
    return pd.read_csv(path, usecols=usecols, **kw)


def build_cohort(mimic, cfg):
    stays = read(mimic, "icu/icustays.csv.gz", ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
                 parse_dates=["intime", "outtime"])
    pats = read(mimic, "hosp/patients.csv.gz", ["subject_id", "gender", "anchor_age"])
    adm = read(mimic, "hosp/admissions.csv.gz", ["hadm_id", "hospital_expire_flag"])
    df = stays.merge(pats, on="subject_id").merge(adm, on="hadm_id")
    c = cfg["data"]["cohort"]
    df = df[(df.anchor_age >= c["min_age"]) & (df.los * 24 >= c["min_icu_los_hours"])]
    df = df.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    return df.reset_index(drop=True)


def load_vitals(mimic, stay_ids):
    out = []
    for ch in pd.read_csv(os.path.join(mimic, "icu/chartevents.csv.gz"),
                          usecols=["stay_id", "charttime", "itemid", "valuenum"], parse_dates=["charttime"], chunksize=5_000_000):
        ch = ch[ch.itemid.isin(VITALS) & ch.stay_id.isin(stay_ids)].dropna(subset=["valuenum"])
        out.append(ch)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_dir", required=True)
    ap.add_argument("--config", default="part2_configs/model_config.yaml")
    ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    d = cfg["data"]
    os.makedirs(a.out, exist_ok=True)

    # ---- 1. cohort ---------------------------------------------------------------------------
    coh = build_cohort(a.mimic_dir, cfg)
    vit = load_vitals(a.mimic_dir, set(coh.stay_id))
    vit = vit.merge(coh[["stay_id", "intime", "outtime"]], on="stay_id")
    vit = vit[(vit.charttime >= vit.intime) & (vit.charttime <= vit.outtime)]
    span = vit.groupby("stay_id").charttime.agg(["min", "max"])
    ok = span[(span["max"] - span["min"]).dt.total_seconds() / 3600 >= d["cohort"]["min_monitoring_hours"]].index
    coh = coh[coh.stay_id.isin(ok)].reset_index(drop=True)
    vit = vit[vit.stay_id.isin(ok)].copy()
    print(f"cohort: {len(coh)} patients")

    # ---- 2. patient-level split --------------------------------------------------------------
    rng = np.random.RandomState(d["split_seed"])
    perm = rng.permutation(len(coh))
    n_tr = int(round(d["split"]["train"] * len(coh)))
    n_va = int(round(d["split"]["val"] * len(coh)))
    coh["split"] = "test"
    coh.loc[perm[:n_tr], "split"] = "train"
    coh.loc[perm[n_tr:n_tr + n_va], "split"] = "val"
    train_stays = set(coh.stay_id[coh.split == "train"])

    # ---- 3. raw event sources ----------------------------------------------------------------
    sid2hadm = dict(zip(coh.stay_id, coh.hadm_id))
    hadm2stay = {v: k for k, v in sid2hadm.items()}
    t0 = dict(zip(coh.stay_id, coh.intime))
    t1 = dict(zip(coh.stay_id, coh.outtime))

    lab = pd.concat([c[c.itemid.isin(LABS) & c.hadm_id.isin(hadm2stay)] for c in pd.read_csv(
        os.path.join(a.mimic_dir, "hosp/labevents.csv.gz"), usecols=["hadm_id", "itemid", "charttime", "valuenum", "flag"],
        parse_dates=["charttime"], chunksize=5_000_000)], ignore_index=True)
    lab["stay_id"] = lab.hadm_id.map(hadm2stay)
    rx = pd.concat([c[c.hadm_id.isin(hadm2stay)] for c in pd.read_csv(
        os.path.join(a.mimic_dir, "hosp/prescriptions.csv.gz"), usecols=["hadm_id", "starttime", "drug"],
        parse_dates=["starttime"], chunksize=2_000_000)], ignore_index=True)
    rx["stay_id"] = rx.hadm_id.map(hadm2stay)
    proc = read(a.mimic_dir, "icu/procedureevents.csv.gz", ["stay_id", "starttime", "itemid"], parse_dates=["starttime"])
    proc = proc[proc.stay_id.isin(sid2hadm)]
    dx = read(a.mimic_dir, "hosp/diagnoses_icd.csv.gz", ["hadm_id", "icd_code", "icd_version"])
    dx = dx[(dx.icd_version == 10) & dx.hadm_id.isin(hadm2stay)].copy()
    dx["code3"] = dx.icd_code.str[:3]

    # ---- 4. fit on TRAIN only ------------------------------------------------------------------
    tv = vit[vit.stay_id.isin(train_stays)]
    vit_stats = {int(i): {"lo": float(g.valuenum.quantile(0.10)), "hi": float(g.valuenum.quantile(0.90)),
                          "median": float(g.valuenum.median())} for i, g in tv.groupby("itemid")}
    top_dx = dx[dx.hadm_id.isin({sid2hadm[s] for s in train_stays})].code3.value_counts().index[:d["n_comorbidity_codes"]].tolist()

    # ---- 5. tokenise each stay ---------------------------------------------------------------------
    ev_rows = []  # (stay_id, hour, token_string)
    # vitals: hourly mean -> ffill -> train-median fill -> category; emit on category change
    vit["hour"] = np.floor((vit.charttime - vit.stay_id.map(t0)).dt.total_seconds() / 3600.0 / d["vital_bin_hours"]).astype(int)
    for (s, i), g in vit.groupby(["stay_id", "itemid"]):
        h = g.groupby("hour").valuenum.mean()
        grid = h.reindex(range(int(h.index.min()), int(h.index.max()) + 1))
        grid = grid.ffill().fillna(vit_stats[int(i)]["median"])
        cat = np.where(grid < vit_stats[int(i)]["lo"], "LOW", np.where(grid > vit_stats[int(i)]["hi"], "HIGH", "NRM"))
        prev = None
        for hr, c in zip(grid.index, cat):
            if c != prev:
                ev_rows.append((s, float(hr * d["vital_bin_hours"]), f"VIT_{VITALS[int(i)]}_{c}"))
                prev = c
    # labs: daily bin, last value of the day
    lab["rel"] = (lab.charttime - lab.stay_id.map(t0)).dt.total_seconds() / 3600.0
    lab = lab[(lab.rel >= 0) & (lab.charttime <= lab.stay_id.map(t1))]
    lab_raw = lab.copy()
    lab["day"] = np.floor(lab.rel / d["lab_bin_hours"]).astype(int)
    lab = lab.sort_values("charttime").groupby(["stay_id", "itemid", "day"]).tail(1)
    for r in lab.itertuples():
        ev_rows.append((r.stay_id, float(r.rel), f"LAB_{LABS[int(r.itemid)]}_{'ABN' if r.flag == 'abnormal' else 'NRM'}"))
    # medications and procedures
    rx["rel"] = (rx.starttime - rx.stay_id.map(t0)).dt.total_seconds() / 3600.0
    for r in rx[(rx.rel >= 0) & (rx.starttime <= rx.stay_id.map(t1))].itertuples():
        ev_rows.append((r.stay_id, float(r.rel), f"RX_{str(r.drug).lower().strip()}"))
    proc["rel"] = (proc.starttime - proc.stay_id.map(t0)).dt.total_seconds() / 3600.0
    for r in proc[(proc.rel >= 0) & (proc.starttime <= proc.stay_id.map(t1))].itertuples():
        ev_rows.append((r.stay_id, float(r.rel), f"PROC_{int(r.itemid)}"))
    ev = pd.DataFrame(ev_rows, columns=["stay_id", "hour", "tok"]).sort_values(["stay_id", "hour", "tok"])

    # vocabulary from TRAIN only (top-K by number of training patients having the token)
    tr_ev = ev[ev.stay_id.isin(train_stays)].drop_duplicates(["stay_id", "tok"])
    vocab_toks = tr_ev.tok.value_counts().index[: d["vocab_size"] - 2].tolist()
    vocab = {"<pad>": PAD, "<eos>": EOS, **{t: i + 2 for i, t in enumerate(vocab_toks)}}

    # baseline features: age, sex, comorbidity multi-hot, first-24h labs (+ missing indicators)
    first_lab = lab_raw[lab_raw.rel <= 24].sort_values("charttime").groupby(["stay_id", "itemid"]).valuenum.first().unstack()
    first_lab = first_lab.reindex(index=coh.stay_id, columns=list(LABS))
    miss = first_lab.isna().astype(np.float32).values
    first_lab_tr = first_lab[coh.split.values == "train"]
    lab_med = first_lab_tr.median().fillna(0.0)
    first_lab = first_lab.fillna(lab_med)
    lab_mu, lab_sd = first_lab_tr.fillna(lab_med).mean(), first_lab_tr.fillna(lab_med).std().fillna(1.0).replace(0, 1)
    labz = ((first_lab - lab_mu) / lab_sd).values.astype(np.float32)
    age = coh.anchor_age.values.astype(np.float32)
    age_mu, age_sd = age[coh.split.values == "train"].mean(), age[coh.split.values == "train"].std()
    sex = (coh.gender == "F").values.astype(np.float32)
    dxm = np.zeros((len(coh), len(top_dx)), np.float32)
    h2row = {h: i for i, h in enumerate(coh.hadm_id)}
    for r in dx[dx.code3.isin(top_dx)].itertuples():
        dxm[h2row[r.hadm_id], top_dx.index(r.code3)] = 1
    x_base = np.concatenate([((age - age_mu) / age_sd)[:, None], sex[:, None], dxm, labz, miss], 1).astype(np.float32)

    # sequences
    L = d["max_len"]
    N = len(coh)
    events = np.zeros((N, L), np.int64)
    dts = np.zeros((N, L), np.float32)
    row = {s: i for i, s in enumerate(coh.stay_id)}
    for s, g_all in ev.groupby("stay_id"):
        g = g_all[g_all.tok.isin(vocab)].head(L - 1)
        i = row[s]
        truncated = len(g_all[g_all.tok.isin(vocab)]) > L - 1
        end_h = float(g.hour.iloc[-1]) if (truncated and len(g)) else (t1[s] - t0[s]).total_seconds() / 3600.0
        toks = [vocab[t] for t in g.tok] + [EOS]
        hrs = np.maximum.accumulate(np.maximum(np.array(list(g.hour) + [end_h], dtype=np.float64), 0.0))
        events[i, : len(toks)] = toks
        dts[i, 0] = hrs[0]
        dts[i, 1: len(toks)] = np.diff(hrs)
    mask = (events != PAD).astype(np.float32)
    artifacts = {"vocab": vocab, "vital_stats": vit_stats, "top_comorbidity_codes": top_dx,
                 "x_base_dim": int(x_base.shape[1]), "vocab_size": len(vocab), "max_len": L}
    tr_mask = (coh.split.values == "train")
    ytr = np.log1p(dts[tr_mask][mask[tr_mask] > 0])
    artifacts["dt_log1p_q05"], artifacts["dt_log1p_q95"] = float(np.quantile(ytr, 0.05)), float(np.quantile(ytr, 0.95))
    json.dump(artifacts, open(os.path.join(a.out, "artifacts.json"), "w"), indent=1)
    for sp in ("train", "val", "test"):
        k = (coh.split.values == sp)
        np.savez_compressed(os.path.join(a.out, f"{sp}.npz"), x_base=x_base[k], events=events[k], dt=dts[k], mask=mask[k],
                            stay_id=coh.stay_id.values[k], mortality=coh.hospital_expire_flag.values[k].astype(np.int8))
        print(sp, int(k.sum()))


if __name__ == "__main__":
    main()
