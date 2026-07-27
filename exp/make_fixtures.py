"""Cross-check fixtures: PyTorch ground truth for the browser runtime to reproduce bit-for-bit-ish.
Covers: every exported model's denoiseVP at scattered (x, lam); analytic GMM; a DDIM run; an FM-Euler run.
"""
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import gmm_ring, vp_alpha_sigma_from_logsnr, fm_t_from_logsnr, logsnr
from distill2d import load_json_model, lam_of_t
from toy2d import OUT, edm_wrap

DEV = 'cpu'  # determinism; these are tiny

def denoise_vp(spec_path, x, lam):
    """Mirror of nn.js denoiseVP, driven by the same JSON weights."""
    spec = json.load(open(spec_path))
    net = load_json_model(spec_path).cpu()
    x = torch.tensor(x, dtype=torch.float32); lam = torch.tensor(lam, dtype=torch.float32).reshape(-1, 1)
    a, s = vp_alpha_sigma_from_logsnr(lam)
    kind = spec['type']
    if kind == 'eps':
        out = net(x, lam)
        return ((x - s * out) / a.clamp(min=1e-6)).detach().numpy()
    if kind == 'edm':
        sd = spec['sigma_data']
        sig = torch.exp(-lam / 2)
        x_ve = x / a.clamp(min=1e-9)
        c_skip = sd**2 / (sig**2 + sd**2)
        c_out = sig * sd / (sig**2 + sd**2).sqrt()
        c_in = 1.0 / (sd**2 + sig**2).sqrt()
        return (c_skip * x_ve + c_out * net(c_in * x_ve, lam)).detach().numpy()
    t = fm_t_from_logsnr(lam)
    x_fm = x * (1 - t) / a.clamp(min=1e-9)
    u = net(x_fm, lam)
    return (x_fm - t * u).detach().numpy()

def main():
    rs = np.random.RandomState(42)
    n = 48
    x = rs.randn(n, 2).astype(np.float32) * 1.4
    lam = rs.uniform(-8, 9, size=n).astype(np.float32)
    fx = {'x': x.tolist(), 'lam': lam.tolist(), 'models': {}}
    for name in ['eps_ddpm', 'x0_edm', 'u_fm', 'u_reflow', 'u_1step']:
        p = os.path.join(OUT, f'model_{name}.json')
        if not os.path.exists(p): continue
        fx['models'][name] = denoise_vp(p, x, lam).round(5).tolist()
        print(f'fixture: {name} ok')
    # analytic gmm
    g = gmm_ring(K=6, radius=1.5, std=0.22)
    a, s = vp_alpha_sigma_from_logsnr(torch.tensor(lam).reshape(-1, 1))
    fx['analytic'] = g.denoise(torch.tensor(x), a, s).numpy().round(5).tolist()
    # one deterministic FM-Euler run on u_fm from fixed starts (t grid mirrors diff.js)
    net = load_json_model(os.path.join(OUT, 'model_u_fm.json')).cpu()
    x1 = torch.tensor(rs.randn(16, 2).astype(np.float32))
    xx = x1.clone()
    T_START, T_END, S = 0.985, 0.002, 24
    ts = torch.linspace(T_START, T_END, S + 1)
    for k in range(S):
        t = ts[k].expand(16, 1)
        u = net(xx, lam_of_t(t))
        xx = xx + (ts[k + 1] - ts[k]) * u
    fx['fme_run'] = {'x1': x1.numpy().round(5).tolist(), 'steps': S,
                     't_start': T_START, 't_end': T_END, 'end': xx.detach().numpy().round(5).tolist()}
    with open(os.path.join(OUT, 'fixtures.json'), 'w') as f:
        json.dump(fx, f)
    print('wrote fixtures.json')

if __name__ == '__main__':
    main()
