"""Bake every data asset the composition needs into classic <script> files (no fetch at render).

Outputs (video/assets/js/):
  mathlib.js     - app/js/{gmm,nn,diff}.js stripped of ESM syntax, exposed as window.HF
  data_models.js - trained 2D model specs + gmm meta + vs_analytic eval curves
  data_mnist.js  - 400 MNIST digits (28x28, 4-bit packed, base64) + labels
  data_drift.js  - S8 exposure-bias toy: 3 REAL trained nets' baked rollout trajectories
"""
import base64, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTD = os.path.join(HERE, 'assets', 'js')
os.makedirs(OUTD, exist_ok=True)

# ------------------------------------------------------------- mathlib bundle
def strip_esm(src):
    src = re.sub(r'^import .*?;\s*$', '', src, flags=re.M)
    src = re.sub(r'^export\s+(async\s+)?(function|const|class|let)', r'\1\2', src, flags=re.M)
    src = re.sub(r'^export\s*\{[^}]*\};?\s*$', '', src, flags=re.M)
    assert 'export' not in src, 'unstripped export remains'
    return src

def build_mathlib():
    parts = ['// bundled from app/js/{gmm,nn,diff}.js — ESM stripped, classic script\n(function(){\n']
    for f in ['gmm.js', 'nn.js', 'diff.js']:
        parts.append(f'// ==== {f} ====\n')
        parts.append(strip_esm(open(os.path.join(ROOT, 'app', 'js', f), encoding='utf-8').read()))
    parts.append('''
window.HF = { MLPModel, vpFromLam, fmTFromLam, sigVEFromLam, b64f32,
  GMM, mulberry32, gauss,
  fmEuler, ddim, edmHeun, runToEnd, cfgDenoise, endpointGap, karrasSigmas,
  vpToFM, fmToVP, veToVP, lamFromT_fm, lamFromSig_ve };
})();\n''')
    open(os.path.join(OUTD, 'mathlib.js'), 'w', encoding='utf-8').write(''.join(parts))
    print('mathlib.js written')

# ------------------------------------------------------------- model data
def build_models():
    models = {}
    for name in ['u_fm', 'u_fm_cond', 'u_reflow', 'u_1step', 'eps_ddpm', 'x0_edm']:
        models[name] = json.load(open(os.path.join(ROOT, 'app', 'data', f'model_{name}.json')))
    res = json.load(open(os.path.join(ROOT, 'app', 'data', 'toy2d_results.json')))
    payload = {
        'models': models,
        'gmm': res['meta']['gmm'],
        'sigma_data': res['meta']['sigma_data'],
        'data_points': res['meta']['data_points'][:1200],
        'lams_eval': res['meta']['lams_eval'],
        'vs_analytic': {k: res[k]['vs_analytic'] for k in ['eps_ddpm', 'x0_edm', 'u_fm']},
        'curve_u_fm': res['u_fm']['curve'][::2],
    }
    s = json.dumps(payload)
    open(os.path.join(OUTD, 'data_models.js'), 'w', encoding='utf-8').write(
        'window.HF_DATA = ' + s + ';\n')
    print(f'data_models.js written ({len(s)//1024} KB)')

# ------------------------------------------------------------- mnist pack
def build_mnist(n=400, seed=0):
    sys.path.insert(0, os.path.join(ROOT, 'exp'))
    from torchvision import datasets
    import torch
    ds = datasets.MNIST(os.path.join(ROOT, 'runs', 'data'), train=True, download=False)
    g = torch.Generator().manual_seed(seed)
    # class-balanced pick
    idxs = []
    targets = ds.targets
    for c in range(10):
        pool = (targets == c).nonzero().flatten()
        sel = pool[torch.randperm(len(pool), generator=g)[:n // 10]]
        idxs += sel.tolist()
    x = ds.data[idxs].float() / 255.0            # (n,28,28) in [0,1]
    y = targets[idxs].tolist()
    # 4-bit pack: two pixels per byte
    q = (x * 15).round().byte().numpy().reshape(len(idxs), -1)   # (n,784) 0..15
    hi, lo = q[:, 0::2], q[:, 1::2]
    packed = (hi << 4 | lo).astype(np.uint8).tobytes()
    b64 = base64.b64encode(packed).decode()
    open(os.path.join(OUTD, 'data_mnist.js'), 'w').write(
        f'window.HF_MNIST = {{"n": {len(idxs)}, "b64": "{b64}", "labels": {json.dumps(y)}}};\n')
    print(f'data_mnist.js written ({len(b64)//1024} KB, {len(idxs)} digits)')

# ------------------------------------------------------------- drift toy (REAL training)
def build_drift():
    import torch, torch.nn as nn
    torch.manual_seed(0)
    R, OMEGA = 3.0, 0.16
    DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

    def true_next(p):  # advance along circle from (projected) current point
        ang = torch.atan2(p[:, 1], p[:, 0]) + OMEGA
        return torch.stack([R * torch.cos(ang), R * torch.sin(ang)], 1)

    def mk_net():
        return nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2)).to(DEV)

    def batch(bs, ctx_noise=0.0, gen=None):
        ang = torch.rand(bs, device=DEV, generator=gen) * 6.2832
        p1 = torch.stack([R * torch.cos(ang), R * torch.sin(ang)], 1)
        p2 = true_next(p1)
        tgt = true_next(p2)
        c1 = p1 + ctx_noise * torch.randn(bs, 2, device=DEV, generator=gen)
        c2 = p2 + ctx_noise * torch.randn(bs, 2, device=DEV, generator=gen)
        return torch.cat([c1, c2], 1), tgt

    def train(net, steps, ctx_noise, gen):
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        for i in range(steps):
            x, t = batch(512, ctx_noise, gen)
            loss = ((net(x) - t) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.item())

    gen = torch.Generator(device=DEV).manual_seed(1)
    A, B = mk_net(), mk_net()
    la = train(A, 4000, 0.0, gen);  print(f'  net A (clean ctx) loss {la:.5f}')
    lb = train(B, 4000, 0.35, gen); print(f'  net B (noisy ctx) loss {lb:.5f}')
    # C: start from A, continue on self-rollout states relabeled by true dynamics
    C = mk_net(); C.load_state_dict(A.state_dict())
    opt = torch.optim.Adam(C.parameters(), lr=1e-3)
    for i in range(2000):
        with torch.no_grad():
            ang = torch.rand(256, device=DEV, generator=gen) * 6.2832
            p1 = torch.stack([R * torch.cos(ang), R * torch.sin(ang)], 1)
            p2 = true_next(p1)
            # roll C forward a few steps to reach its OWN state distribution
            a, b = p1, p2
            for _ in range(int(torch.randint(1, 8, (1,), generator=gen, device=DEV))):
                nxt = C(torch.cat([a, b], 1))
                a, b = b, nxt
            tgt = true_next(b)  # relabel with true dynamics from wherever C ended up
        x = torch.cat([a, b], 1)
        loss = ((C(x) - tgt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f'  net C (self-rollout relabel) loss {float(loss.item()):.5f}')

    # rollouts with injected per-step error
    def rollout(net, regime, frames=260, seed=5):
        g2 = torch.Generator(device=DEV).manual_seed(seed)
        a = torch.tensor([[R, 0.0]], device=DEV)
        b = true_next(a)
        pts, drift = [], []
        for k in range(frames):
            nxt = net(torch.cat([a, b], 1))
            if regime == 'radial':
                nrm = nxt / nxt.norm(dim=1, keepdim=True).clamp(min=1e-6)
                nxt = nxt + 0.02 * nrm
            elif regime == 'jitter':
                nxt = nxt + 0.06 * torch.randn(1, 2, device=DEV, generator=g2)
            pts.append([round(float(nxt[0, 0]), 3), round(float(nxt[0, 1]), 3)])
            drift.append(round(abs(float(nxt.norm()) - R), 3))
            a, b = b, nxt
        return pts, drift

    out = {'R': R}
    with torch.no_grad():
        for name, net in [('A', A), ('B', B), ('C', C)]:
            out[name] = {}
            for regime in ['none', 'radial', 'jitter']:
                pts, drift = rollout(net, regime)
                out[name][regime] = {'pts': pts, 'drift': drift}
    for name in 'ABC':
        print(f"  drift@250 radial {name}: {out[name]['radial']['drift'][250]}")
    open(os.path.join(OUTD, 'data_drift.js'), 'w').write('window.HF_DRIFT = ' + json.dumps(out) + ';\n')
    print('data_drift.js written')

if __name__ == '__main__':
    build_mathlib()
    build_models()
    build_mnist()
    build_drift()
    print('ALL EXPORTS DONE')
