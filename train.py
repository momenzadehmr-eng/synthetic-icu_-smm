"""Main code - training / generation driver.

    python -m main.train --config part2_configs/model_config.yaml --data data/processed \
        --seed 11 --out runs/seed11 [--generate 5000]

DP-SGD (manuscript Sec. 2.5): Poisson sampling with expected batch size 64, processed as physical
batches of 16 accumulated over 4 steps; per-sample clipping C = 1.0; Gaussian noise sigma = 1.1; RDP accountant.
BOTH networks that touch real data (generator side and discriminator) are trained with DP-SGD and attached to the
SAME PrivacyEngine, so the accountant is charged for both. Training stops before epsilon would exceed
dp.target_epsilon.

NOT TESTED against the reported numbers: this driver was written from the manuscript and has not been run on
MIMIC-IV by the person who wrote it. Run the smoke test (part4_reproduction) first.
"""
import argparse, json, os, random
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from main.dgsmm import DGSMM, Discriminator
from part3_dp_accounting.privacy_accounting import epsilon as dp_epsilon


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def load_split(path):
    d = np.load(path)
    return (torch.tensor(d["x_base"]).float(), torch.tensor(d["events"]).long(),
            torch.tensor(d["dt"]).float(), torch.tensor(d["mask"]).float())


def load_rules(path, vocab):
    r = json.load(open(path))
    def idx(pairs):
        out = []
        for a, b in pairs:
            if a in vocab and b in vocab:
                out.append((vocab[a], vocab[b]))
            else:
                print(f"[rules] skipping pair with token not in vocabulary: {a}, {b}")
        return out
    return {"guideline_idx": idx(r.get("guideline_forbidden_pairs", [])),
            "order_idx": idx(r.get("order_pairs_first_before_second", []))}


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); tot, n = 0.0, 0
    for xb, ev, dt, m in loader:
        out = model(xb.to(dev), ev.to(dev), dt.to(dev), m.to(dev), None)
        tot += out["nelbo"].item() * xb.shape[0]; n += xb.shape[0]
    return tot / max(n, 1)


def run(cfg, data_dir, seed, out_dir, n_generate=0):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)
    T, DP, CL = cfg["train"], cfg["dp"], cfg["loss"]
    dev = torch.device(T["device"] if torch.cuda.is_available() else "cpu")
    art = json.load(open(os.path.join(data_dir, "artifacts.json")))
    V, Db = art["vocab_size"], art["x_base_dim"]
    tr, va = load_split(os.path.join(data_dir, "train.npz")), load_split(os.path.join(data_dir, "val.npz"))
    rules = load_rules(cfg["rules_file"], art["vocab"])
    dt_band = (art["dt_log1p_q05"], art["dt_log1p_q95"])
    phys = T["physical_batch_size"]; logical = phys * T["accumulation_steps"]; N = len(tr[0])

    model, disc = DGSMM(cfg, Db, V).to(dev), Discriminator(cfg, V).to(dev)
    opt_g = torch.optim.Adam(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    opt_d = torch.optim.Adam(disc.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(*tr), batch_size=logical if DP["enabled"] else phys, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(*va), batch_size=64)

    engine = None
    if DP["enabled"]:
        from opacus import PrivacyEngine
        engine = PrivacyEngine(accountant=DP["accountant"])
        kw = dict(noise_multiplier=DP["noise_multiplier"], max_grad_norm=DP["max_grad_norm"], poisson_sampling=True)
        model, opt_g, dp_loader = engine.make_private(module=model, optimizer=opt_g, data_loader=loader, **kw)
        disc, opt_d, _ = engine.make_private(module=disc, optimizer=opt_d, data_loader=loader, **kw)
        train_loader = dp_loader
    else:
        train_loader = loader

    best, bad, steps, stop, log, micro = float("inf"), 0, 0, False, [], 0
    acc = T["accumulation_steps"]
    for epoch in range(T["epochs"]):
        tau = T["gumbel_tau_start"] * (T["gumbel_tau_end"] / T["gumbel_tau_start"]) ** (epoch / max(1, T["epochs"] - 1))
        model.train(); disc.train()
        for batch in train_loader:
            if batch[0].shape[0] == 0:
                continue
            if DP["enabled"] and dp_epsilon(DP["noise_multiplier"], N, logical, steps + 1, DP["delta"], mechanisms_per_step=2) > DP["target_epsilon"]:
                stop = True
                break
            chunks = list(zip(*[b.split(phys) for b in batch])) if DP["enabled"] else [batch]
            for i, (xb, ev, dt, m) in enumerate(chunks):
                last = i == len(chunks) - 1
                if DP["enabled"]:
                    opt_g.signal_skip_step(do_skip=not last); opt_d.signal_skip_step(do_skip=not last)
                xb, ev, dt, m = xb.to(dev), ev.to(dev), dt.to(dev), m.to(dev)
                out = model(xb, ev, dt, m, rules, tau_gs=tau, cfg_loss=CL, dt_band=dt_band)
                y_real = torch.log1p(dt)
                # ---- discriminator step (real vs. detached generator output) ----
                for p in disc.parameters(): p.requires_grad_(True)
                d_real = disc(F.one_hot(ev, V).float(), y_real, m)
                d_fake = disc(out["fake_probs"].detach(), out["fake_y"].detach(), m)
                loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                         F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
                do_zero = DP["enabled"] or micro % acc == 0          # DP: Opacus handles virtual steps itself
                do_step = DP["enabled"] or (micro + 1) % acc == 0
                sc = 1.0 if DP["enabled"] else 1.0 / acc               # non-DP: plain gradient accumulation
                if do_zero: opt_d.zero_grad()
                (loss_d * sc).backward()
                if do_step: opt_d.step()
                # ---- generator step: -ELBO + lambda_adv*L_adv + L_clinical ----
                for p in disc.parameters(): p.requires_grad_(False)       # keep D out of G's per-sample grads
                d_g = disc(out["fake_probs"], out["fake_y"], m)
                loss_g = out["nelbo"] + CL["lambda_adv"] * F.binary_cross_entropy_with_logits(d_g, torch.ones_like(d_g)) \
                         + CL["lambda_guideline"] * out["l_guideline"] + CL["lambda_order"] * out["l_order"] \
                         + CL["lambda_temporal"] * out["l_temporal"]
                if do_zero: opt_g.zero_grad()
                (loss_g * sc).backward()
                if do_step: opt_g.step()
                micro += 1
            steps += 1
        val = evaluate(model, val_loader, dev)
        eps = float(engine.get_epsilon(DP["delta"])) if engine else None
        log.append({"epoch": epoch, "steps": steps, "val_nelbo": val, "epsilon": eps})
        print(json.dumps(log[-1]))
        if val < best - 1e-4:
            best, bad = val, 0
            torch.save((model._module if engine else model).state_dict(), os.path.join(out_dir, "best.pt"))
        else:
            bad += 1
        if stop or bad >= T["early_stopping_patience"]:
            break

    final = {"seed": seed, "epochs_run": len(log), "optimizer_steps": steps, "best_val_nelbo": best,
             "epsilon_reported_by_opacus": log[-1]["epsilon"], "delta": DP["delta"] if engine else None,
             "stopped_by_budget": stop, "mechanisms_charged_per_step": 2 if engine else 0}
    json.dump({"summary": final, "log": log}, open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)

    if n_generate:
        net = DGSMM(cfg, Db, V).to(dev)
        net.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=dev))
        evs, dts, ms = [], [], []
        for s in range(0, n_generate, 256):
            e, d_, k = net.sample(min(256, n_generate - s), art["max_len"], dev)
            evs.append(e.cpu()); dts.append(d_.cpu()); ms.append(k.cpu())
        np.savez_compressed(os.path.join(out_dir, "synthetic.npz"), events=torch.cat(evs).numpy(),
                            dt=torch.cat(dts).numpy(), mask=torch.cat(ms).numpy())
    return final


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="part2_configs/model_config.yaml")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generate", type=int, default=0)
    ap.add_argument("--no_dp", action="store_true", help="disable DP-SGD (non-private ablation)")
    ap.add_argument("--max_epochs", type=int, default=None, help="override for smoke tests")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.no_dp: cfg["dp"]["enabled"] = False
    if a.max_epochs: cfg["train"]["epochs"] = a.max_epochs
    print(run(cfg, a.data, a.seed, a.out, a.generate))
