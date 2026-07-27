"""Render every MNIST asset the lab needs, from the trained checkpoints.
Outputs -> app/data/mnist/  (PNG montages + JSON stats).
"""
import json, math, os, sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from mnist import UNet, load_mnist, sd_linear_alphabar, RUNS
from common import logsnr

DEV = 'cuda'
OUT = os.path.join(os.path.dirname(__file__), '..', 'app', 'data', 'mnist')
os.makedirs(OUT, exist_ok=True)
GAP, GAPV = 2, 32  # montage gap px and gray value

def load(tag):
    ck = torch.load(os.path.join(RUNS, f'{tag}.pt'), map_location=DEV, weights_only=False)
    net = UNet(b=ck['b']).to(DEV); net.load_state_dict(ck['ema']); net.eval()
    return net, ck

def to_u8(x, lo=-1.0, hi=1.0):
    return ((x.clamp(lo, hi) - lo) / (hi - lo) * 255).byte().cpu().numpy()

def montage(cells, rows, cols, path):
    """cells: list of HxW uint8 arrays, row-major."""
    H, W = cells[0].shape
    img = np.full((rows * H + (rows - 1) * GAP, cols * W + (cols - 1) * GAP), GAPV, dtype=np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        img[r * (H + GAP):r * (H + GAP) + H, k * (W + GAP):k * (W + GAP) + W] = c
    Image.fromarray(img).save(path)
    print(f'  wrote {os.path.relpath(path)} ({rows}x{cols})')

# ---- FM-frame denoiser + samplers (torch mirror of the browser code) ----

def lam_of_t(t): return 2 * torch.log((1 - t) / t)

class D:
    """denoiseVP(x_vp, lam) -> x0_hat, wrapping the FM-native UNet."""
    def __init__(self, net, w=1.0, cls=None):
        self.net, self.w, self.cls = net, w, cls
    def raw_u(self, x_fm, lam, cls):
        B = x_fm.shape[0]
        c = torch.full((B,), cls, device=DEV, dtype=torch.long)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self.net(x_fm, lam.reshape(B, 1), c).float()
    def x0_fm(self, x_fm, lam):
        t = torch.sigmoid(-lam / 2).reshape(-1, 1, 1, 1)
        u = self.raw_u(x_fm, lam, self.cls if self.cls is not None else 10)
        if self.cls is not None and self.w != 1.0:
            un = self.raw_u(x_fm, lam, 10)
            u = un + self.w * (u - un)
        return x_fm - t * u
    def vp(self, x_vp, lam):
        a = torch.sigmoid(lam).sqrt().reshape(-1, 1, 1, 1)
        t = torch.sigmoid(-lam / 2).reshape(-1, 1, 1, 1)
        return self.x0_fm(x_vp * (1 - t) / a.clamp(min=1e-8), lam)

@torch.no_grad()
def fme(d, x1, steps, record=None):
    x = x1.clone()
    ts = torch.linspace(0.985, 0.002, steps + 1, device=DEV)
    for k in range(steps):
        t = ts[k]; lam = lam_of_t(t.expand(x.shape[0]))
        x0h = d.x0_fm(x, lam)
        u = (x - x0h) / t
        if record is not None: record.append((float(t), x.clone(), x0h.clone(), u.clone()))
        x = x + (ts[k + 1] - ts[k]) * u
    return x

@torch.no_grad()
def ddim_vp(d, xT, steps, eta=0.0, lam_lo=-8.5, lam_hi=9.0, gen=None):
    x = xT.clone()
    lams = torch.linspace(lam_lo, lam_hi, steps + 1, device=DEV)
    for k in range(steps):
        lam, lamn = lams[k], lams[k + 1]
        a, s = torch.sigmoid(lam).sqrt(), torch.sigmoid(-lam).sqrt()
        an, sn = torch.sigmoid(lamn).sqrt(), torch.sigmoid(-lamn).sqrt()
        x0h = d.vp(x, lam.expand(x.shape[0]))
        eps = (x - a * x0h) / s
        tau = eta * (sn / s) * (1 - (a / an) ** 2).clamp(min=0).sqrt()
        tau = torch.minimum(tau, sn)
        x = an * x0h + (sn ** 2 - tau ** 2).clamp(min=0).sqrt() * eps
        if tau > 0:
            x = x + tau * torch.randn(x.shape, device=DEV, generator=gen)
    return x

@torch.no_grad()
def heun_ve(d, x_ve, steps, lam_lo=-8.5, lam_hi=9.0, rho=7.0):
    smax, smin = math.exp(-lam_lo / 2), math.exp(-lam_hi / 2)
    i = torch.arange(steps + 1, device=DEV) / steps
    sig = (smax ** (1 / rho) + i * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    sig[-1] = 0
    x = x_ve.clone()
    for k in range(steps):
        s, sn = sig[k], sig[k + 1]
        lam = (-2 * torch.log(s)).expand(x.shape[0])
        a = torch.sigmoid(lam[0]).sqrt()
        x0h = d.vp(x * a, lam)
        dd = (x - x0h) / s
        xn = x + (sn - s) * dd
        if sn > 0:
            lam2 = (-2 * torch.log(sn)).expand(x.shape[0])
            a2 = torch.sigmoid(lam2[0]).sqrt()
            x0h2 = d.vp(xn * a2, lam2)
            dd2 = (xn - x0h2) / sn
            xn = x + (sn - s) * 0.5 * (dd + dd2)
        x = xn
    return x

@torch.no_grad()
def ddpm_native_leaky(net, n, steps=250, gen=None, cheat_init=None):
    """Ancestral sampling with the leaky model's OWN discrete schedule, from N(0,I)."""
    betas, ab = sd_linear_alphabar()
    betas, ab = betas.to(DEV), ab.to(DEV)
    idx = torch.linspace(999, 0, steps).long().to(DEV)
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=gen) if cheat_init is None else cheat_init
    cls = torch.randint(0, 10, (n,), device=DEV, generator=gen)
    for j in range(steps):
        ti = idx[j]
        a_bar = ab[ti]; a_bar_prev = ab[idx[j + 1]] if j + 1 < steps else torch.tensor(1.0, device=DEV)
        a, s = a_bar.sqrt(), (1 - a_bar).sqrt()
        lam = torch.log(a_bar / (1 - a_bar)).expand(n)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            eps = net(x, lam.reshape(n, 1), cls).float()
        x0h = ((x - s * eps) / a).clamp(-1, 1)
        # posterior q(x_{t-1} | x_t, x0)
        ap, sp = a_bar_prev.sqrt(), (1 - a_bar_prev).sqrt()
        beta_t = 1 - a_bar / a_bar_prev
        mu = (ap * beta_t / (1 - a_bar)) * x0h + ((a_bar / a_bar_prev).sqrt() * (1 - a_bar_prev) / (1 - a_bar)) * x
        if j + 1 < steps:
            var = beta_t * (1 - a_bar_prev) / (1 - a_bar)
            x = mu + var.sqrt() * torch.randn(x.shape, device=DEV, generator=gen)
        else:
            x = mu
    return x

# ---------------------------------------------------------------- exports

def export_anatomy(net):
    x_all, y_all = load_mnist()
    x0 = x_all[7:8]  # a crisp digit
    g = torch.Generator(device=DEV).manual_seed(3)
    eps = torch.randn(1, 1, 32, 32, device=DEV, generator=g)
    LAMS = [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 9]
    d = D(net, cls=int(y_all[7].item()))
    rows = {k: [] for k in ['x_t', 'x0h', 'epsh', 'vh', 'uh', 'score']}
    stats = []
    for L in LAMS:
        lam = torch.tensor([float(L)], device=DEV)
        t = float(torch.sigmoid(-lam / 2))
        a, s = float(torch.sigmoid(lam).sqrt()), float(torch.sigmoid(-lam).sqrt())
        x_t = (1 - t) * x0 + t * eps                     # FM frame (what the net sees)
        x0h = d.x0_fm(x_t, lam)
        epsh = (x_t - (1 - t) * x0h) / max(t, 1e-6)
        vh = a * epsh - s * x0h
        uh = epsh - x0h
        score = -epsh / s
        cell = {'lam': L, 't': t, 'alpha_vp': a, 'sigma_vp': s}
        for key, ten, rng in [('x_t', x_t, 1.6), ('x0h', x0h, 1.0), ('epsh', epsh, 3.0),
                              ('vh', vh, 2.0), ('uh', uh, 3.0), ('score', score, None)]:
            v = ten[0, 0]
            sc = float(v.abs().max()) if rng is None else rng
            rows[key].append(to_u8(v, -sc, sc))
            cell[key] = {'std': float(v.std()), 'absmax': float(v.abs().max()), 'disp_range': sc}
        cell['x0h_mse'] = float(((x0h - x0) ** 2).mean())
        stats.append(cell)
    cells = [c for k in ['x_t', 'x0h', 'epsh', 'vh', 'uh', 'score'] for c in rows[k]]
    montage(cells, 6, len(LAMS), os.path.join(OUT, 'anatomy.png'))
    Image.fromarray(to_u8(x0[0, 0])).save(os.path.join(OUT, 'anatomy_x0.png'))
    Image.fromarray(to_u8(eps[0, 0], -3, 3)).save(os.path.join(OUT, 'anatomy_eps.png'))
    json.dump({'lams': LAMS, 'stats': stats, 'digit': int(y_all[7].item()),
               'data_pixel_std': float(x_all[:4096].std()), 'data_pixel_mean': float(x_all[:4096].mean())},
              open(os.path.join(OUT, 'anatomy.json'), 'w'))

def export_infer_trace(net):
    g = torch.Generator(device=DEV).manual_seed(11)
    x1 = torch.randn(1, 1, 32, 32, device=DEV, generator=g)
    d = D(net, cls=5)
    rec = []
    fme(d, x1, 8, record=rec)
    cells, stats = [], []
    for (t, x, x0h, u) in rec:
        cells.append(to_u8(x[0, 0], -1.6, 1.6))
        stats.append({'t': t, 'lam': float(2 * math.log((1 - t) / t)),
                      'x_std': float(x.std()), 'x0h_std': float(x0h.std()), 'u_rms': float(u.pow(2).mean().sqrt())})
    for (_, _, x0h, _) in rec: cells.append(to_u8(x0h[0, 0], -1, 1))
    for (_, _, _, u) in rec: cells.append(to_u8(u[0, 0], -3, 3))
    montage(cells, 3, len(rec), os.path.join(OUT, 'infer_trace.png'))
    json.dump(stats, open(os.path.join(OUT, 'infer_trace.json'), 'w'))

def export_sampler_grids(net):
    meta = {}
    for name in ['fme', 'ddim', 'ddpm', 'heun']:
        for nfe in [1, 2, 4, 8, 16, 32]:
            g = torch.Generator(device=DEV).manual_seed(5150)
            x1 = torch.randn(64, 1, 32, 32, device=DEV, generator=g)
            cls = (torch.arange(64, device=DEV) // 8) % 10
            outs = []
            for c in range(8):  # batch by row to vary class cleanly
                d = D(net, cls=int(cls[c * 8].item()))
                xr = x1[c * 8:(c + 1) * 8]
                if name == 'fme': o = fme(d, xr, nfe)
                elif name == 'ddim': o = ddim_vp(d, xr, nfe, eta=0)
                elif name == 'ddpm': o = ddim_vp(d, xr, nfe, eta=1, gen=g)
                else: o = heun_ve(d, xr * math.exp(8.5 / 2), max(1, nfe // 2))
                outs.append(o)
            o = torch.cat(outs)
            cellso = [to_u8(o[i, 0]) for i in range(64)]
            montage(cellso, 8, 8, os.path.join(OUT, f'grid_{name}_{nfe}.png'))
            meta[f'{name}_{nfe}'] = {'nfe_actual': nfe if name != 'heun' else max(1, nfe // 2) * 2}
    json.dump(meta, open(os.path.join(OUT, 'grids.json'), 'w'))

def export_cfg(net):
    stats = []
    for w in [0, 1, 1.5, 2, 3, 5, 8, 14]:
        g = torch.Generator(device=DEV).manual_seed(2024)
        x1 = torch.randn(32, 1, 32, 32, device=DEV, generator=g)
        outs = []
        for r, digit in enumerate([0, 3, 5, 8]):
            d = D(net, w=w, cls=digit) if w > 0 else D(net, cls=None)
            outs.append(fme(d, x1[r * 8:(r + 1) * 8], 32))
        o = torch.cat(outs)
        montage([to_u8(o[i, 0]) for i in range(32)], 4, 8, os.path.join(OUT, f'cfg_{str(w).replace(".", "p")}.png'))
        stats.append({'w': w, 'sat': float((o.abs() > 0.95).float().mean()),
                      'std': float(o.std()), 'mean': float(o.mean())})
    json.dump(stats, open(os.path.join(OUT, 'cfg.json'), 'w'))

def export_museum(net_fm, net_leaky):
    g = torch.Generator(device=DEV).manual_seed(31337)
    # healthy fm samples
    x1 = torch.randn(64, 1, 32, 32, device=DEV, generator=g)
    cls = (torch.arange(64, device=DEV) // 8) % 10
    outs = []
    for c in range(8):
        outs.append(fme(D(net_fm, cls=int(cls[c * 8].item())), x1[c * 8:(c + 1) * 8], 32))
    fm_o = torch.cat(outs)
    montage([to_u8(fm_o[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'museum_fm.png'))
    # leaky, honest init from N(0,I)
    leaky_o = ddpm_native_leaky(net_leaky, 64, gen=g)
    montage([to_u8(leaky_o[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'museum_leaky.png'))
    # leaky, forensic init: x_T drawn WITH the data leak it was trained to expect
    x_all, _ = load_mnist()
    pick = x_all[torch.randint(0, x_all.shape[0], (64,), device=DEV, generator=g)]
    _, ab = sd_linear_alphabar()
    aT, sT = ab[-1].sqrt().to(DEV), (1 - ab[-1]).sqrt().to(DEV)
    cheat = aT * pick + sT * torch.randn(64, 1, 32, 32, device=DEV, generator=g)
    leaky_fix = ddpm_native_leaky(net_leaky, 64, gen=g, cheat_init=cheat)
    montage([to_u8(leaky_fix[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'museum_leaky_fixedinit.png'))
    # histograms
    def hist(t):
        h = torch.histc(t.flatten().float().cpu(), bins=64, min=-1.05, max=1.05)
        return (h / h.sum()).tolist()
    data_batch = x_all[:2048]
    json.dump({
        'hist_bins': 64, 'hist_range': [-1.05, 1.05],
        'hist_data': hist(data_batch), 'hist_fm': hist(fm_o), 'hist_leaky': hist(leaky_o),
        'hist_leaky_fixed': hist(leaky_fix),
        'mean_data': float(data_batch.mean()), 'mean_fm': float(fm_o.mean()), 'mean_leaky': float(leaky_o.mean()),
        'terminal_sqrt_abar': float(aT), 'terminal_logsnr': float(torch.log(ab[-1] / (1 - ab[-1]))),
    }, open(os.path.join(OUT, 'museum.json'), 'w'))

def export_student(net_fm, net_student):
    g = torch.Generator(device=DEV).manual_seed(808)
    x1 = torch.randn(32, 1, 32, 32, device=DEV, generator=g)
    cls = (torch.arange(32, device=DEV) // 8) % 10
    t_outs, s_outs = [], []
    for c in range(4):
        xr = x1[c * 8:(c + 1) * 8]
        t_outs.append(fme(D(net_fm, cls=int(cls[c * 8].item())), xr, 32))
        # student: one shot; trained convention x0 = x1 - span * mean-velocity, lam at t=0.985
        span = 0.985 - 0.002
        lam1 = lam_of_t(torch.tensor(0.985, device=DEV)).expand(8)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            u = net_student(xr, lam1.reshape(8, 1), torch.full((8,), int(cls[c * 8].item()), device=DEV, dtype=torch.long)).float()
        s_outs.append(xr - span * u)
    t_o, s_o = torch.cat(t_outs), torch.cat(s_outs)
    montage([to_u8(t_o[i, 0]) for i in range(32)], 4, 8, os.path.join(OUT, 'student_teacher.png'))
    montage([to_u8(s_o[i, 0]) for i in range(32)], 4, 8, os.path.join(OUT, 'student_1step.png'))
    lap = lambda t: float(torch.var(t[:, :, 1:, :] - t[:, :, :-1, :]) + torch.var(t[:, :, :, 1:] - t[:, :, :, :-1]))
    json.dump({'hf_teacher': lap(t_o), 'hf_student': lap(s_o), 'teacher_nfe': 32, 'student_nfe': 1},
              open(os.path.join(OUT, 'student.json'), 'w'))

@torch.no_grad()
def ddim_native_leaky(net, n, steps=50, gen=None, cheat_init=None):
    """Deterministic DDIM (eta=0) on the leaky model's own discrete schedule.
    No fresh noise anywhere: all global structure must come from x_T."""
    _, ab = sd_linear_alphabar(); ab = ab.to(DEV)
    idx = torch.linspace(999, 0, steps + 1).long().to(DEV)
    x = torch.randn(n, 1, 32, 32, device=DEV, generator=gen) if cheat_init is None else cheat_init
    cls = torch.randint(0, 10, (n,), device=DEV, generator=gen)
    for j in range(steps):
        a_bar, a_next = ab[idx[j]], (ab[idx[j + 1]] if j + 1 < steps else torch.tensor(1.0, device=DEV))
        a, s = a_bar.sqrt(), (1 - a_bar).sqrt()
        an, sn = a_next.sqrt(), (1 - a_next).sqrt()
        lam = torch.log(a_bar / (1 - a_bar)).expand(n)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            eps = net(x, lam.reshape(n, 1), cls).float()
        x0h = ((x - s * eps) / a).clamp(-1, 1)
        x = an * x0h + sn * eps
    return x

def export_museum_lum(net_fm, net_leaky):
    """Luminance-diverse MNIST: where the terminal-SNR bug actually bites.
    Metric: per-image background level (10th percentile pixel)."""
    g = torch.Generator(device=DEV).manual_seed(4242)
    x_all, _ = load_mnist()
    beta = torch.rand(2048, 1, 1, 1, device=DEV, generator=g)
    data = (x_all[:2048] + beta).clamp(-1, 1)

    def bg(t): return torch.quantile(t.float().flatten(1), 0.1, dim=1)

    # healthy FM samples
    x1 = torch.randn(256, 1, 32, 32, device=DEV, generator=g)
    outs = []
    for i in range(8):
        outs.append(fme(D(net_fm, cls=i), x1[i * 32:(i + 1) * 32], 32))
    fm_o = torch.cat(outs)
    # leaky, honest init: ancestral (stochastic) AND ddim (deterministic)
    leaky_anc = ddpm_native_leaky(net_leaky, 256, gen=g)
    leaky_det = ddim_native_leaky(net_leaky, 256, steps=50, gen=g)
    # forensic init for the deterministic case (leak carried from data with full beta range)
    pick = (x_all[torch.randint(0, x_all.shape[0], (256,), device=DEV, generator=g)]
            + torch.rand(256, 1, 1, 1, device=DEV, generator=g)).clamp(-1, 1)
    _, ab = sd_linear_alphabar()
    aT, sT = ab[-1].sqrt().to(DEV), (1 - ab[-1]).sqrt().to(DEV)
    cheat = aT * pick + sT * torch.randn(256, 1, 32, 32, device=DEV, generator=g)
    leaky_fix = ddim_native_leaky(net_leaky, 256, steps=50, gen=g, cheat_init=cheat)

    montage([to_u8(data[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'lum_data.png'))
    montage([to_u8(fm_o[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'lum_fm.png'))
    montage([to_u8(leaky_det[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'lum_leaky.png'))
    montage([to_u8(leaky_fix[i, 0]) for i in range(64)], 8, 8, os.path.join(OUT, 'lum_leaky_fixedinit.png'))

    def hist(t, bins=32):
        h = torch.histc(bg(t).cpu(), bins=bins, min=-1.05, max=0.3)
        return (h / h.sum()).tolist()
    json.dump({
        'bins': 32, 'range': [-1.05, 0.3],
        'bg_data': hist(data), 'bg_fm': hist(fm_o), 'bg_leaky_anc': hist(leaky_anc),
        'bg_leaky': hist(leaky_det), 'bg_cheat': hist(leaky_fix),
        'std_data': float(bg(data).std()), 'std_fm': float(bg(fm_o).std()),
        'std_leaky_anc': float(bg(leaky_anc).std()), 'std_leaky': float(bg(leaky_det).std()),
        'std_cheat': float(bg(leaky_fix).std()),
    }, open(os.path.join(OUT, 'museum_lum.json'), 'w'))
    print('  lum museum done')

def export_curves(cks):
    json.dump({k: ck['curve'] for k, ck in cks.items()}, open(os.path.join(OUT, 'curves.json'), 'w'))

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fm, ck_fm = load('mnist_fm')
    print('anatomy...'); export_anatomy(fm)
    print('infer trace...'); export_infer_trace(fm)
    print('sampler grids...'); export_sampler_grids(fm)
    print('cfg...'); export_cfg(fm)
    leaky, ck_leaky = load('mnist_leaky')
    print('museum...'); export_museum(fm, leaky)
    try:
        student, _ = load('mnist_1step')
        print('student...'); export_student(fm, student)
    except FileNotFoundError:
        print('student checkpoint not ready, skipping')
    export_curves({'mnist_fm': ck_fm, 'mnist_leaky': ck_leaky})
    print('DONE')
