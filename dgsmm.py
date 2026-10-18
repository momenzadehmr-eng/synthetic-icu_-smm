"""Main code - DGSMM model (Deep Generative Semi-Markov Model).

Reference implementation written from the manuscript's Methods (Sections 2.2-2.3).
Opacus-compatible: batch-first tensors, DPGRU instead of nn.GRU, hand-written attention
built from nn.Linear (no nn.MultiheadAttention, no BatchNorm).

Data tensors (all batch-first):
    x_base (B, Db)  baseline features        ev (B, L) long  event tokens, 0 = pad, 1 = <EOS>
    dt     (B, L)   hours since previous event (first: since ICU admission)
    m      (B, L)   1.0 for real positions

IMPLEMENTATION CHOICES the manuscript does not pin down (document them in the paper if they match yours):
  * Latent states s_l are inferred per event step by a recurrent recognition network + straight-through
    Gumbel-Softmax (Sec. 2.2.6). A "run" is a maximal block of equal consecutive hard states.
  * Transition NLL is evaluated at run starts only (self-transitions are masked: persistence is
    governed by the sojourn-time model). The first state uses a patient-conditioned initial distribution.
  * Sojourn NLL: at each run end (state changes at l+1) the MDN scores log1p(dwell), dwell = sum of
    inter-event times inside the run, conditioned on (s_l, s_{l+1}, z0, h_l). Runs cut by the end of the
    sequence are treated as censored and skipped.
  * Generation: at run entry draw s_next ~ transition (no self-loop), tau ~ MDN; event times inside the
    run are drawn by the temporal network and clipped so the run's total never exceeds tau.
  * L_guideline / L_order operate on soft (probability) event presence; rules come from clinical_rules.json.
  * L_temporal: only the quantile-band penalty is implemented. The vital-sign smoothness term mentioned in
    the manuscript needs continuous vitals, which the tokenised representation does not contain.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPGRU


def mlp(i, h, o, n_layers, p):
    layers, d = [], i
    for _ in range(n_layers - 1):
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        assert d % heads == 0
        self.h, self.p = heads, p
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        B, L, D = x.shape
        sp = lambda t: t.view(B, L, self.h, D // self.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sp(self.q(x)), sp(self.k(x)), sp(self.v(x)), is_causal=True,
                                           dropout_p=self.p if self.training else 0.0)
        return self.o(a.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, d, heads, p):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = CausalSelfAttention(d, heads, p)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = x + self.drop(self.att(self.n1(x)))
        return x + self.drop(self.ff(self.n2(x)))


class DGSMM(nn.Module):
    """Generator side: encoder, semi-Markov module, event generator, temporal network."""

    def __init__(self, cfg, x_dim, vocab):
        super().__init__()
        m, S = cfg["model"], cfg["data"]["n_states"]
        d, zd, H, p = m["embed_dim"], m["latent_dim"], m["history_gru"]["hidden"], m["dropout"]
        self.S, self.V, self.K, self.zd = S, vocab, m["duration_mdn_components"], zd
        self.enc = mlp(x_dim, m["encoder"]["hidden"], 2 * zd, m["encoder"]["layers"], p)      # 3-layer MLP, 256
        self.ev_emb = nn.Embedding(vocab, d, padding_idx=0)
        self.t_proj = nn.Linear(1, d)
        self.z_proj = nn.Linear(zd, d)
        self.s_proj_h = nn.Linear(S, d, bias=False)
        self.s_proj_e = nn.Linear(S, d, bias=False)
        # recognition network q(s_l | events<=l, z0)  and history network h_<l
        self.rec_gru = DPGRU(d, H, num_layers=m["history_gru"]["layers"], batch_first=True)
        self.rec_head = nn.Linear(H, S)
        self.hist_gru = DPGRU(d, H, num_layers=m["history_gru"]["layers"], batch_first=True)
        # semi-Markov: transition, initial state, duration MDN
        self.trans = mlp(S + zd + H, 128, S, 2, p)
        self.init = nn.Linear(zd, S)
        self.dur = mlp(2 * S + zd + H, 128, 3 * self.K, 2, p)
        # clinical event generator (attention)
        self.blocks = nn.ModuleList([Block(d, m["event_transformer"]["heads"], p) for _ in range(m["event_transformer"]["layers"])])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        # temporal-coordination network  P(dt | e_{l-1}, e_l, s_l)
        self.tnet = mlp(2 * d + S, 128, 2, 2, p)

    # ---- helpers ------------------------------------------------------------------------------
    @staticmethod
    def shift(x):                       # shift right by one along sequence dim, zero at BOS
        return F.pad(x, (0, 0, 1, 0))[:, :-1]

    def event_logits(self, prev_step, s_onehot, zp):
        h = prev_step + self.s_proj_e(s_onehot) + zp.unsqueeze(1)
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))

    def mdn_params(self, inp):
        o = self.dur(inp)
        lp, mu, ls = o.split(self.K, dim=-1)
        return F.log_softmax(lp, -1), mu, ls.clamp(-5, 3)

    # ---- training forward (teacher forcing) -------------------------------------------------------
    def forward(self, x_base, ev, dt, m, rules, tau_gs=1.0, cfg_loss=None, dt_band=None):
        B, L = ev.shape
        m = m.float()
        n = m.sum(1).clamp(min=1.0)
        y = torch.log1p(dt)
        mu, logvar = self.enc(x_base).chunk(2, -1)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        zp = self.z_proj(z)
        e_emb = self.ev_emb(ev)
        step_in = e_emb + self.t_proj(y.unsqueeze(-1))

        # q(s | .) with straight-through Gumbel-Softmax
        q_logits = self.rec_head(self.rec_gru(step_in + zp.unsqueeze(1))[0])
        s_st = F.gumbel_softmax(q_logits, tau=tau_gs, hard=True, dim=-1)
        q_logp = F.log_softmax(q_logits, -1)
        neg_ent = (q_logp.exp() * q_logp).sum(-1)
        s_idx = s_st.argmax(-1)

        # history h_<l (uses previous event, previous state, z0)
        prev_s = self.shift(s_st)
        h = self.hist_gru(self.shift(step_in) + self.s_proj_h(prev_s) + zp.unsqueeze(1))[0]
        zx = z.unsqueeze(1).expand(-1, L, -1)

        # transition NLL at run starts
        tr = self.trans(torch.cat([prev_s, zx, h], -1)).masked_fill(prev_s > 0.5, -1e4)
        tr = torch.cat([self.init(z).unsqueeze(1), tr[:, 1:]], 1)
        nll_tr = -(s_st * F.log_softmax(tr, -1)).sum(-1)
        change = torch.ones_like(m, dtype=torch.bool)
        change[:, 1:] = s_idx[:, 1:] != s_idx[:, :-1]
        nll_tr = (nll_tr * change.float() * m).sum(1)

        # sojourn-time MDN NLL at run ends
        dwell, dws = torch.zeros(B, device=dt.device), []
        for l in range(L - 1):
            dwell = dwell + dt[:, l]
            dws.append(dwell)
            dwell = torch.where(change[:, l + 1], torch.zeros_like(dwell), dwell)
        dws = torch.stack(dws, 1)                                           # (B, L-1)
        run_end = (change[:, 1:] & (m[:, 1:] > 0)).float()
        lp, mu_d, ls_d = self.mdn_params(torch.cat([s_st[:, :-1], s_st[:, 1:], zx[:, :-1], h[:, :-1]], -1))
        yd = torch.log1p(dws).unsqueeze(-1)
        comp = lp - 0.5 * ((yd - mu_d) / ls_d.exp()) ** 2 - ls_d - 0.5 * math.log(2 * math.pi)
        nll_dur = (-(torch.logsumexp(comp, -1)) * run_end).sum(1)

        # event generator
        logits = self.event_logits(self.shift(step_in), s_st, zp)
        ce = F.cross_entropy(logits.reshape(-1, self.V), ev.reshape(-1), reduction="none").view(B, L)
        nll_ev = (ce * m).sum(1)

        # temporal network
        t_in = torch.cat([self.shift(e_emb), e_emb, s_st], -1)
        mu_t, ls_t = self.tnet(t_in).chunk(2, -1)
        mu_t, ls_t = mu_t.squeeze(-1), ls_t.squeeze(-1).clamp(-5, 3)
        nll_t = ((0.5 * ((y - mu_t) / ls_t.exp()) ** 2 + ls_t + 0.5 * math.log(2 * math.pi)) * m).sum(1)

        kl_z = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1)
        nelbo = ((nll_ev + nll_t + nll_tr + nll_dur + (neg_ent * m).sum(1) + kl_z) / n).mean()

        # soft "fake" trajectory for the discriminator / clinical terms
        probs = F.softmax(logits, -1)
        y_fake = mu_t + ls_t.exp() * torch.randn_like(mu_t)
        out = {"nelbo": nelbo, "fake_probs": probs, "fake_y": y_fake, "m": m,
               "parts": {"ev": nll_ev.mean() / n.mean(), "time": nll_t.mean() / n.mean(),
                         "trans": nll_tr.mean() / n.mean(), "dur": nll_dur.mean() / n.mean(), "kl_z": kl_z.mean()}}
        out.update(self.clinical_losses(probs, y_fake, m, rules, cfg_loss, dt_band))
        return out

    # ---- clinical constraint regularisers (Sec. 2.3.3) ---------------------------------------------
    def clinical_losses(self, probs, y_fake, m, rules, cfg_loss, dt_band):
        z = probs.new_zeros(())
        out = {"l_guideline": z, "l_order": z, "l_temporal": z}
        if rules is None or cfg_loss is None:
            return out
        def p_any(tok):
            return 1.0 - torch.exp((torch.log1p(-probs[:, :, tok].clamp(max=1 - 1e-6)) * m).sum(1))
        pairs = rules.get("guideline_idx", [])
        if pairs:
            out["l_guideline"] = sum((p_any(a) * p_any(b)).mean() for a, b in pairs)
        T = torch.cumsum(torch.expm1(y_fake).clamp(min=0) * m, 1)
        def t_exp(tok):
            w = probs[:, :, tok] * m
            return (w * T).sum(1) / (w.sum(1) + 1e-6)
        order = rules.get("order_idx", [])      # (first, second): `first` must precede `second`
        if order:
            eps = cfg_loss["order_eps_hours"]
            out["l_order"] = sum((p_any(a) * p_any(b) * F.relu(t_exp(a) - t_exp(b) + eps)).mean() for a, b in order)
        if dt_band is not None:
            lo, hi = dt_band
            out["l_temporal"] = (((F.relu(lo - y_fake) + F.relu(y_fake - hi)) * m).sum(1) / m.sum(1).clamp(min=1.0)).mean()   # per-sample mean (DP-SGD needs mean-of-per-sample losses)
        return out

    # ---- ancestral sampling (generation time) ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, n, max_len, device):
        self.eval()
        z = torch.randn(n, self.zd, device=device)
        zp = self.z_proj(z)
        s = torch.distributions.Categorical(logits=self.init(z)).sample()
        eye = torch.eye(self.S, device=device)
        step_prev = torch.zeros(n, self.ev_emb.embedding_dim, device=device)
        s_prev_oh = torch.zeros(n, self.S, device=device)
        hid = None
        need = torch.ones(n, dtype=torch.bool, device=device)
        s_next = s.clone(); remaining = torch.zeros(n, device=device)
        steps_in, states, toks, dts = [], [], [], []
        alive = torch.ones(n, dtype=torch.bool, device=device)
        for l in range(max_len):
            g_in = (step_prev + self.s_proj_h(s_prev_oh) + zp).unsqueeze(1)
            ho, hid = self.hist_gru(g_in, hid) if hid is not None else self.hist_gru(g_in)
            h_l = ho[:, 0]
            s_oh = eye[s]
            if need.any():                        # run entry: draw next state and sojourn time
                lg = self.trans(torch.cat([s_oh, z, h_l], -1)).masked_fill(s_oh > 0.5, -1e4)
                s_new_next = torch.distributions.Categorical(logits=lg).sample()
                lp, mu_d, ls_d = self.mdn_params(torch.cat([s_oh, eye[s_new_next], z, h_l], -1))
                k = torch.distributions.Categorical(logits=lp).sample().unsqueeze(-1)
                tau = torch.expm1(mu_d.gather(-1, k).squeeze(-1) + ls_d.gather(-1, k).squeeze(-1).exp() * torch.randn(n, device=device)).clamp(min=0)
                s_next = torch.where(need, s_new_next, s_next)
                remaining = torch.where(need, tau, remaining)
            states.append(s_oh)
            sts = torch.stack(states, 1)
            prefix = torch.stack([torch.zeros_like(step_prev)] + steps_in, 1) if steps_in else torch.zeros_like(step_prev).unsqueeze(1)
            lg_e = self.event_logits(prefix, sts, zp)[:, -1]
            e = torch.distributions.Categorical(logits=lg_e).sample()
            e_emb = self.ev_emb(e)
            mu_t, ls_t = self.tnet(torch.cat([self.ev_emb(toks[-1]) if toks else torch.zeros_like(e_emb), e_emb, s_oh], -1)).chunk(2, -1)
            y = mu_t.squeeze(-1) + ls_t.squeeze(-1).clamp(-5, 3).exp() * torch.randn(n, device=device)
            dt = torch.expm1(y).clamp(min=0)
            dt_c = torch.minimum(dt, remaining)
            remaining = remaining - dt_c
            toks.append(e); dts.append(torch.where(alive, dt_c, torch.zeros_like(dt_c)))
            step_prev = e_emb + self.t_proj(torch.log1p(dt_c).unsqueeze(-1))
            steps_in.append(step_prev)
            s_prev_oh = s_oh
            leave = remaining <= 1e-6
            s = torch.where(leave, s_next, s)
            need = leave
            alive = alive & (e != 1)
        ev = torch.stack(toks, 1)
        eos = (ev == 1).float().cumsum(1)
        keep = ((eos - (ev == 1).float()) == 0).float()          # keep up to and including first <EOS>
        return ev * keep.long(), torch.stack(dts, 1) * keep, keep


class ResConv(nn.Module):
    def __init__(self, c, k, p):
        super().__init__()
        self.c = nn.Conv1d(c, c, k, padding=k // 2)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.c(F.leaky_relu(x, 0.2)))


class Discriminator(nn.Module):
    """4-layer residual 1-D conv discriminator over (event distribution, log1p dt) sequences."""

    def __init__(self, cfg, vocab):
        super().__init__()
        c, k, p = cfg["model"]["discriminator"]["channels"], cfg["model"]["discriminator"]["kernel"], cfg["model"]["dropout"]
        self.inp, self.tin = nn.Linear(vocab, c), nn.Linear(1, c)
        self.convs = nn.ModuleList([ResConv(c, k, p) for _ in range(cfg["model"]["discriminator"]["conv_layers"])])
        self.out = nn.Linear(c, 1)

    def forward(self, ev_dist, y, m):
        x = (self.inp(ev_dist) + self.tin(y.unsqueeze(-1))).transpose(1, 2)
        for cv in self.convs:
            x = cv(x)
        x = (x.transpose(1, 2) * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return self.out(F.leaky_relu(x, 0.2)).squeeze(-1)
