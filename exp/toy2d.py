"""Suite A: 2D toy models, trained natively in each lineage's dialect, exported for the JS lab.

Models (all condition on logSNR via Fourier features; heads/weighting/time-sampling differ):
  eps_ddpm : eps-prediction, VP-cosine, t ~ U[0,1]          (DDPM 2020 dialect)
  x0_edm   : EDM preconditioned D, lognormal sigma, EDM w   (Karras 2022 dialect)
  u_fm     : FM linear velocity, t ~ logit-normal(0,1)      (SD3/Flux 2024 dialect)
  u_fm_cond: same + class label w/ 10% dropout              (for CFG)
  u_fm_moons: FM on two-moons (curved manifold visuals)

Everything evaluated against the ANALYTIC denoiser of the GMM (closed form), so
model error vs irreducible floor is exactly separable.
"""
import json, math, base64, os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from common import (gmm_ring, GMM, vp_cosine_alpha_sigma, fm_alpha_sigma, logsnr,
                    vp_alpha_sigma_from_logsnr, fm_t_from_logsnr)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT = os.path.join(os.path.dirname(__file__), '..', 'app', 'data')
os.makedirs(OUT, exist_ok=True)

LAM_SCALE = 0.25
FREQS = [0.5, 1.0, 2.0, 4.0, 8.0]

def lam_feats(lam):
    l = lam * LAM_SCALE
    out = [l]
    for f in FREQS:
        out += [torch.sin(f * l), torch.cos(f * l)]
    return torch.cat(out, dim=-1)  # (B, 11)

class MLP(nn.Module):
    def __init__(self, n_class=0, emb_dim=8, hidden=128):
        super().__init__()
        self.n_class = n_class
        in_dim = 2 + 11 + (emb_dim if n_class else 0)
        if n_class:
            self.emb = nn.Embedding(n_class + 1, emb_dim)  # last index = null token
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))
    def forward(self, x, lam, cls=None):
        h = [x, lam_feats(lam)]
        if self.n_class:
            h.append(self.emb(cls))
        return self.net(torch.cat(h, dim=-1))

# ---------------------------------------------------------------- dialects

def make_batch_ddpm(x0, gen):
    """VP cosine, t~U[0,1], target = eps, loss = ||eps_hat - eps||^2."""
    B = x0.shape[0]
    t = torch.rand(B, 1, device=x0.device, generator=gen)
    a, s = vp_cosine_alpha_sigma(t)
    eps = torch.randn(B, 2, device=x0.device, generator=gen)
    x_t = a * x0 + s * eps
    lam = logsnr(a, s)
    return x_t, lam, eps, torch.ones(B, device=x0.device), ('eps', a, s)

def make_batch_edm(x0, gen, sigma_data, P_mean, P_std):
    """VE frame, ln sigma ~ N(P_mean, P_std), EDM weighting on D-space loss."""
    B = x0.shape[0]
    sig = torch.exp(P_mean + P_std * torch.randn(B, 1, device=x0.device, generator=gen))
    eps = torch.randn(B, 2, device=x0.device, generator=gen)
    x_t = x0 + sig * eps
    lam = logsnr(torch.ones_like(sig), sig)
    w = (sig[:, 0]**2 + sigma_data**2) / (sig[:, 0] * sigma_data)**2
    return x_t, lam, x0, w, ('edm', torch.ones_like(sig), sig)

def make_batch_fm(x0, gen):
    """Linear path, t ~ logit-normal(0,1), target u = eps - x0."""
    B = x0.shape[0]
    t = torch.sigmoid(torch.randn(B, 1, device=x0.device, generator=gen))
    eps = torch.randn(B, 2, device=x0.device, generator=gen)
    x_t = (1 - t) * x0 + t * eps
    lam = logsnr(1 - t, t)
    return x_t, lam, eps - x0, torch.ones(B, device=x0.device), ('u', 1 - t, t)

def edm_wrap(net, x, lam, sigma_data, cls=None):
    """EDM preconditioning: D = c_skip*x + c_out*F(c_in*x, .)."""
    sig = torch.exp(-lam / 2)
    c_skip = sigma_data**2 / (sig**2 + sigma_data**2)
    c_out = sig * sigma_data / (sig**2 + sigma_data**2).sqrt()
    c_in = 1.0 / (sigma_data**2 + sig**2).sqrt()
    return c_skip * x + c_out * net(c_in * x, lam, cls)

def to_x0(kind, out, x_t, a, s):
    """Native head output -> x0_hat (the ONE object)."""
    if kind == 'eps': return (x_t - s * out) / a.clamp(min=1e-6)
    if kind == 'edm': return out                       # already D = x0_hat
    if kind == 'u':   return x_t - s * out             # FM linear: t == s here (alpha=1-t)
    raise ValueError(kind)

# ---------------------------------------------------------------- training

N_BINS, LAM_LO, LAM_HI = 24, -9.0, 10.0

def train(name, dialect, data_sampler, n_class=0, steps=25000, bs=1024, lr=1e-3, seed=0,
          gmm=None, sigma_data=None):
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed + 1)
    net = MLP(n_class=n_class).to(DEV)
    ema = MLP(n_class=n_class).to(DEV); ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    bin_native = torch.zeros(N_BINS, device=DEV); bin_x0 = torch.zeros(N_BINS, device=DEV)
    bin_cnt = torch.zeros(N_BINS, device=DEV)
    curve = []  # (step, mean_native_loss, per-bin x0 loss snapshot)
    for it in range(steps):
        x0, cls = data_sampler(bs, gen)
        if dialect == 'ddpm':
            x_t, lam, target, w, (kind, a, s) = make_batch_ddpm(x0, gen)
            out = net(x_t, lam, cls)
        elif dialect == 'edm':
            x_t, lam, target, w, (kind, a, s) = make_batch_edm(x0, gen, sigma_data, P_mean=-0.4, P_std=1.2)
            out = edm_wrap(net, x_t, lam, sigma_data, cls)
        elif dialect == 'fm':
            x_t, lam, target, w, (kind, a, s) = make_batch_fm(x0, gen)
            out = net(x_t, lam, cls)
        per = ((out - target)**2).sum(-1)
        loss = (w * per).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for p_e, p in zip(ema.parameters(), net.parameters()):
                p_e.mul_(0.9995).add_(p, alpha=0.0005)
            for b_e, b in zip(ema.buffers(), net.buffers()):
                b_e.copy_(b)
            # per-logSNR bookkeeping in BOTH native and x0 space (GPU, no sync)
            x0h = to_x0(kind, out, x_t, a, s)
            per_x0 = ((x0h - x0)**2).sum(-1)
            bins = ((lam[:, 0] - LAM_LO) / (LAM_HI - LAM_LO) * N_BINS).long().clamp(0, N_BINS - 1)
            bin_native.mul_(0.99).index_add_(0, bins, 0.01 * per.detach())
            bin_x0.mul_(0.99).index_add_(0, bins, 0.01 * per_x0.detach())
            bin_cnt.mul_(0.99).index_add_(0, bins, 0.01 * torch.ones_like(per_x0))
        if it % 500 == 0 or it == steps - 1:
            cnt = bin_cnt.cpu().numpy(); bx0 = bin_x0.cpu().numpy()
            snap = np.where(cnt > 1e-8, bx0 / np.maximum(cnt, 1e-12), np.nan)
            curve.append({'step': it, 'loss': float(loss.item()),
                          'x0_per_bin': [None if np.isnan(v) else float(v) for v in snap]})
            if it % 5000 == 0:
                print(f'  [{name}] step {it:6d}  native loss {loss.item():.4f}')
    return ema, curve

def eval_vs_analytic(net_denoise, gmm, lams, n=8192, seed=123):
    """Model error E||D_theta - D*||^2 and floor E[Var] per lam. net_denoise(x,lam)->x0_hat."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    rows = []
    for L in lams:
        lam = torch.full((n, 1), float(L), device=DEV)
        a, s = vp_alpha_sigma_from_logsnr(lam)
        x0, _ = gmm.sample(n, generator=gen)
        x_t = a * x0 + s * torch.randn(n, 2, device=DEV, generator=gen)
        with torch.no_grad():
            dstar = gmm.denoise(x_t, a, s)
            dhat = net_denoise(x_t, lam)
            err = ((dhat - dstar)**2).sum(-1).mean().item()
            floor = gmm.x0_posterior_var(x_t, a, s).mean().item()
            mse = ((dhat - x0)**2).sum(-1).mean().item()
        rows.append({'lam': float(L), 'model_err': err, 'floor': floor, 'total_mse': mse})
    return rows

# ---------------------------------------------------------------- export

def b64f32(t):
    a = t.detach().cpu().float().numpy().astype('<f4')
    return base64.b64encode(a.tobytes()).decode()

def export_model(net, kind, path, sigma_data=None, n_class=0, extra=None):
    layers = []
    for m in net.net:
        if isinstance(m, nn.Linear):
            layers.append({'w': b64f32(m.weight), 'b': b64f32(m.bias), 'shape': list(m.weight.shape)})
    obj = {'type': kind, 'lam_scale': LAM_SCALE, 'freqs': FREQS,
           'hidden': 128, 'layers': layers, 'n_class': n_class,
           'sigma_data': sigma_data}
    if n_class:
        obj['emb'] = b64f32(net.emb.weight)
        obj['emb_dim'] = net.emb.weight.shape[1]
    if extra: obj.update(extra)
    with open(path, 'w') as f: json.dump(obj, f)
    print(f'  exported {path} ({os.path.getsize(path)//1024} KB)')

def moons(n, gen, noise=0.06):
    t = torch.rand(n, 1, device=DEV, generator=gen) * math.pi
    top = torch.rand(n, 1, device=DEV, generator=gen) < 0.5
    x = torch.where(top, torch.cat([torch.cos(t), torch.sin(t)], 1),
                    torch.cat([1 - torch.cos(t), -torch.sin(t) + 0.5], 1))
    x = x + noise * torch.randn(n, 2, device=DEV, generator=gen)
    x = (x - torch.tensor([[0.5, 0.25]], device=DEV)) * 1.4
    return x, None

def main():
    gmm = gmm_ring(K=6, radius=1.5, std=0.22, device=DEV)
    xs, _ = gmm.sample(200000, generator=torch.Generator(device=DEV).manual_seed(9))
    sigma_data = xs.std().item()
    print(f'sigma_data(gmm6) = {sigma_data:.4f}')

    def samp_gmm(bs, gen):
        x, c = gmm.sample(bs, generator=gen); return x, c
    def samp_gmm_uncond(bs, gen):
        x, _ = gmm.sample(bs, generator=gen); return x, None
    def samp_gmm_cfg(bs, gen):
        x, c = gmm.sample(bs, generator=gen)
        drop = torch.rand(bs, device=DEV, generator=gen) < 0.10
        c = torch.where(drop, torch.full_like(c, 6), c)
        return x, c
    def samp_moons(bs, gen):
        return moons(bs, gen)

    lams_eval = np.linspace(-8, 9, 35)
    results = {}

    jobs = [
        ('eps_ddpm', 'ddpm', samp_gmm_uncond, 0, None),
        ('x0_edm',   'edm',  samp_gmm_uncond, 0, sigma_data),
        ('u_fm',     'fm',   samp_gmm_uncond, 0, None),
        ('u_fm_cond','fm',   samp_gmm_cfg,    7, None),
        ('u_fm_moons','fm',  samp_moons,      0, None),
    ]
    nets = {}
    for name, dialect, sampler, n_class, sd in jobs:
        print(f'training {name} ({dialect}) ...')
        net, curve = train(name, dialect, sampler, n_class=n_class, gmm=gmm, sigma_data=sd)
        nets[name] = (net, dialect, n_class, sd)
        results[name] = {'curve': curve}
        export_model(net, {'ddpm': 'eps', 'edm': 'edm', 'fm': 'u'}[dialect],
                     os.path.join(OUT, f'model_{name}.json'), sigma_data=sd, n_class=n_class)

    # denoise adapters (native head -> x0_hat), for eval
    def adapter(name):
        net, dialect, n_class, sd = nets[name]
        def d(x, lam):
            cls = torch.full((x.shape[0],), 6, device=DEV, dtype=torch.long) if n_class else None
            if dialect == 'edm':
                return edm_wrap(net, x, lam, sd, cls)
            out = net(x, lam, cls)
            if dialect == 'ddpm':
                a, s = vp_alpha_sigma_from_logsnr(lam)
                return (x - s * out) / a.clamp(min=1e-6)
            t = fm_t_from_logsnr(lam)
            return x - t * out
        return d

    print('evaluating vs analytic ground truth ...')
    for name in ['eps_ddpm', 'x0_edm', 'u_fm']:
        results[name]['vs_analytic'] = eval_vs_analytic(adapter(name), gmm, lams_eval)

    # pairwise field disagreement per lam (are they "the same field"?)
    print('pairwise field agreement ...')
    gen = torch.Generator(device=DEV).manual_seed(77)
    pair_rows = []
    for L in lams_eval:
        lam = torch.full((8192, 1), float(L), device=DEV)
        a, s = vp_alpha_sigma_from_logsnr(lam)
        x0, _ = gmm.sample(8192, generator=gen)
        x_t = a * x0 + s * torch.randn(8192, 2, device=DEV, generator=gen)
        with torch.no_grad():
            d = {n: adapter(n)(x_t, lam) for n in ['eps_ddpm', 'x0_edm', 'u_fm']}
            dstar = gmm.denoise(x_t, a, s)
        row = {'lam': float(L), 'scale': float((dstar**2).sum(-1).mean().item())}
        for i, j in [('eps_ddpm', 'x0_edm'), ('eps_ddpm', 'u_fm'), ('x0_edm', 'u_fm')]:
            row[f'{i}|{j}'] = float(((d[i] - d[j])**2).sum(-1).mean().item())
        pair_rows.append(row)
    results['pairwise'] = pair_rows

    # data scatter + gmm spec for the app (JS re-implements the analytic denoiser)
    pts, cls = gmm.sample(2500, generator=torch.Generator(device=DEV).manual_seed(5))
    mpts, _ = moons(1500, torch.Generator(device=DEV).manual_seed(6))
    results['meta'] = {
        'sigma_data': sigma_data,
        'gmm': {'mu': gmm.mu.cpu().tolist(), 'std': gmm.std.cpu().tolist(), 'w': gmm.w.cpu().tolist()},
        'data_points': pts.cpu().numpy().round(4).tolist(),
        'data_labels': cls.cpu().tolist(),
        'moons_points': mpts.cpu().numpy().round(4).tolist(),
        'lams_eval': [float(x) for x in lams_eval],
    }
    with open(os.path.join(OUT, 'toy2d_results.json'), 'w') as f:
        json.dump(results, f)
    print('wrote toy2d_results.json')

    # console summary: model error vs floor at a few lams (goes into the narrative)
    print('\n=== model error E||D_theta - D*||^2 (floor = E[posterior var]) ===')
    print(f'{"lam":>6} {"floor":>9} {"eps_ddpm":>9} {"x0_edm":>9} {"u_fm":>9}')
    for i, L in enumerate(lams_eval):
        if i % 4: continue
        r = {n: results[n]['vs_analytic'][i] for n in ['eps_ddpm', 'x0_edm', 'u_fm']}
        print(f'{L:6.1f} {r["u_fm"]["floor"]:9.4f} ' + ' '.join(f'{r[n]["model_err"]:9.5f}' for n in ['eps_ddpm', 'x0_edm', 'u_fm']))

if __name__ == '__main__':
    main()
