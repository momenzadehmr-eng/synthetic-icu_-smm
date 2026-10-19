# DGSMM: Deep Generative Semi-Markov Model for synthetic ICU patient trajectories

Code accompanying the manuscript. The package is organised as
**one main code package and four supporting parts**.

| Component | Folder | Contents |
|---|---|---|
| **Main code** | `main/` | `dgsmm.py` (encoder, semi-Markov module, event generator, temporal network, discriminator, losses, sampler); `train.py` (DP-SGD training, early stopping, generation) |
| **Part 1: Preprocessing** | `part1_preprocessing/` | `preprocess_mimic4.py`: cohort selection, patient-level split, hourly/daily alignment, tokenisation, imputation (fitted on the training split only) |
| **Part 2: Configurations and seeds** | `part2_configs/` | `model_config.yaml`, `seeds.json` (10 runs), `clinical_rules.json` |
| **Part 3: DP accounting** | `part3_dp_accounting/` | `privacy_accounting.py`: RDP accounting, budget search, stand-alone epsilon check |
| **Part 4: Reproduction** | `part4_reproduction/` | `REPRODUCE.md`, `requirements.txt`, `run_all.sh`, fake-data smoke test |

## Quick start

```bash
pip install -r part4_reproduction/requirements.txt

# 1) smoke test on FAKE random data (no MIMIC access needed)
bash part4_reproduction/tests/smoke_test.sh

# 2) full pipeline on MIMIC-IV (requires credentialed PhysioNet access)
bash part4_reproduction/run_all.sh /path/to/mimiciv/3.1
```

Privacy accounting only (no GPU needed):

```bash
python part3_dp_accounting/privacy_accounting.py --sigma 1.1 --n_train 13120 --batch 64 --steps 12500 15000 --target_epsε ≤ 3.1 
```

## Outputs
`data/processed/{train,val,test}.npz` and `artifacts.json` (vocabulary, fitted statistics);
`runs/seed<k>/best.pt`, `train_log.json` (epochs, steps, epsilon), `synthetic.npz`.

## Data
MIMIC-IV is not distributed here. Obtain it from PhysioNet under its data use agreement.

## Status and known limitations
- Reference implementation written from the manuscript's Methods.
- MIMIC item ids in `preprocess_mimic4.py` should be verified against `d_items` / `d_labitems`.
- Training stops before epsilon would exceed `dp.target_epsilon`. Generator and discriminator are both trained with DP-SGD on one `PrivacyEngine`, so each step is charged twice; with sigma = 1.1 and q = 64/13120 this reaches epsilon ≤ 3.1  after about 4,250 steps (about 21 epochs), not 12,500-15,000.
- Evaluation code (fidelity, privacy attacks, downstream utility) and baselines are not included.
- Early stopping uses the validation split without DP accounting.

## License
Add the licence of your choice before public release.
