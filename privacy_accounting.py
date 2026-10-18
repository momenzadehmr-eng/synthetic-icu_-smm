"""Part 3 - differential-privacy accounting routines.

DP-SGD with Poisson sampling, per-sample clipping C and Gaussian noise sigma; privacy loss tracked with
the Renyi-DP (RDP) accountant (Opacus) and converted to (epsilon, delta)-DP.

CLI examples
    # what epsilon do the manuscript's settings give?
    python part3_dp_accounting/privacy_accounting.py --sigma 1.1 --n_train 13120 --batch 64 \
        --steps 12500 15000 --delta 1e-5
    # how many steps fit in a budget / what sigma is needed
    python part3_dp_accounting/privacy_accounting.py --sigma 1.1 --n_train 13120 --batch 64 --target_eps 2.3

IMPORTANT - number of mechanisms per step:
    If two networks are trained on the same real minibatch with DP-SGD at the same (sigma, q) -
    here the generator side AND the discriminator - each logical step releases TWO noisy gradients, and the
    accountant must be charged twice (mechanisms_per_step=2). Opacus does this automatically when both
    optimizers are attached to the same PrivacyEngine (see main/train.py).
"""
import argparse


def _rdp_epsilon(sigma, q, steps, delta):
    """epsilon after `steps` Poisson-subsampled Gaussian mechanisms (RDP accountant)."""
    try:
        from opacus.accountants import RDPAccountant
        acc = RDPAccountant()
        acc.history = [(sigma, q, int(steps))]
        return float(acc.get_epsilon(delta))
    except ImportError:
        from dp_accounting import dp_event                      # pip install dp-accounting
        from dp_accounting.rdp import RdpAccountant
        acc = RdpAccountant()
        acc.compose(dp_event.PoissonSampledDpEvent(q, dp_event.GaussianDpEvent(sigma)), int(steps))
        return float(acc.get_epsilon(delta))


def epsilon(sigma, n_train, batch, steps, delta=1e-5, mechanisms_per_step=1):
    q = batch / n_train
    return _rdp_epsilon(sigma, q, steps * mechanisms_per_step, delta)


def max_steps_for_budget(sigma, n_train, batch, target_eps, delta=1e-5, mechanisms_per_step=1, hi=2_000_000):
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if epsilon(sigma, n_train, batch, mid, delta, mechanisms_per_step) <= target_eps:
            lo = mid
        else:
            hi = mid - 1
    return lo


def sigma_for_budget(steps, n_train, batch, target_eps, delta=1e-5, mechanisms_per_step=1):
    lo, hi = 0.3, 50.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if epsilon(mid, n_train, batch, steps, delta, mechanisms_per_step) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi


class BudgetGuard:
    """Use inside the training loop: stop BEFORE the budget is exceeded.

        guard = BudgetGuard(privacy_engine, delta=1e-5, target_eps=2.3)
        ... after each optimizer step:  if guard.exhausted(): break
    """

    def __init__(self, privacy_engine, delta, target_eps):
        self.engine, self.delta, self.target = privacy_engine, delta, target_eps

    def current_epsilon(self):
        return float(self.engine.get_epsilon(self.delta))

    def exhausted(self):
        return self.current_epsilon() >= self.target


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.1)
    ap.add_argument("--n_train", type=int, default=13120)
    ap.add_argument("--batch", type=int, default=64, help="expected (logical) batch size")
    ap.add_argument("--steps", type=int, nargs="*", default=[])
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--target_eps", type=float, default=None)
    a = ap.parse_args()
    print(f"q = {a.batch / a.n_train:.5f}, sigma = {a.sigma}, delta = {a.delta}")
    for st in a.steps:
        for k in (1, 2):
            print(f"steps={st:>6}  mechanisms/step={k}  epsilon={epsilon(a.sigma, a.n_train, a.batch, st, a.delta, k):.3f}")
    if a.target_eps:
        for k in (1, 2):
            ms = max_steps_for_budget(a.sigma, a.n_train, a.batch, a.target_eps, a.delta, k)
            print(f"mechanisms/step={k}: max steps with eps<={a.target_eps}: {ms} (~{ms * a.batch / a.n_train:.1f} epochs)")
            for st in a.steps:
                print(f"   sigma needed for eps<={a.target_eps} at {st} steps: {sigma_for_budget(st, a.n_train, a.batch, a.target_eps, a.delta, k):.2f}")
