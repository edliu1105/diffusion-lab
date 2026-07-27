"""Suite B: MNIST models.

  mnist_fm    : class-conditional flow matching UNet (10% label dropout -> CFG),
                logit-normal t. The healthy modern baseline.
  mnist_leaky : eps-prediction + SD-style scaled-linear beta schedule => non-zero
                terminal SNR (sqrt(alphabar_T) ~ 0.04 leak). Trained faithfully,
                sampled from N(0,I) => the classic washed-out/gray failure, reproduced.
  mnist_1step : naive distillation student, regresses (noise -> teacher ODE endpoint).

Canvas is 32x32 (MNIST padded with background = -1). Data range [-1, 1].
All models condition on logSNR (lambda); per-lambda loss bins logged like toy2d.
"""
import math, os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import logsnr, vp_alpha_sigma_from_logsnr

DEV = 'cuda'
ROOT = os.path.join(os.path.dirname(__file__), '..')
RUNS = os.path.join(ROOT, 'runs'); os.makedirs(RUNS, exist_ok=True)

# ------------------------------------------------------------------ data

def load_mnist():
    from torchvision import datasets
    ds = datasets.MNIST(os.path.join(RUNS, 'data'), train=True, download=True)
    x = (ds.data.float() / 255.0) * 2 - 1                       # (60000,28,28) in [-1,1]
    x = F.pad(x, (2, 2, 2, 2), value=-1.0).unsqueeze(1)         # (60000,1,32,32)
    y = ds.targets.clone()
    return x.to(DEV), y.to(DEV)

# ------------------------------------------------------------------ tiny UNet

class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin); self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(tdim, cout * 2)
        self.n2 = nn.GroupNorm(8, cout); self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(self, x, t):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(t))[:, :, None, None].chunk(2, dim=1)
        h = self.c2(F.silu(self.n2(h) * (1 + scale) + shift))
        return h + self.skip(x)

class Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n = nn.GroupNorm(8, c); self.qkv = nn.Conv2d(c, c * 3, 1); self.out = nn.Conv2d(c, c, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        a = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        return x + self.out((v @ a.transpose(1, 2)).reshape(B, C, H, W))

class UNet(nn.Module):
    """32x32, base ch b, mults (1,2,2), attn at 8x8. ~8M params at b=64."""
    def __init__(self, b=64, tdim=256, n_class=11):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.cemb = nn.Embedding(n_class, tdim)
        self.cin = nn.Conv2d(1, b, 3, padding=1)
        self.d1a, self.d1b = ResBlock(b, b, tdim), ResBlock(b, b, tdim)
        self.down1 = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)       # 32->16
        self.d2a, self.d2b = ResBlock(b * 2, b * 2, tdim), ResBlock(b * 2, b * 2, tdim)
        self.down2 = nn.Conv2d(b * 2, b * 2, 3, stride=2, padding=1)   # 16->8
        self.d3a, self.d3b = ResBlock(b * 2, b * 2, tdim), ResBlock(b * 2, b * 2, tdim)
        self.attn3 = Attn(b * 2)
        self.mid1, self.mida, self.mid2 = ResBlock(b * 2, b * 2, tdim), Attn(b * 2), ResBlock(b * 2, b * 2, tdim)
        self.u3a = ResBlock(b * 4, b * 2, tdim); self.attnu3 = Attn(b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b * 2, 4, stride=2, padding=1)  # 8->16
        self.u2a = ResBlock(b * 4, b * 2, tdim)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)      # 16->32
        self.u1a = ResBlock(b * 2, b, tdim)
        self.out_n = nn.GroupNorm(8, b); self.out_c = nn.Conv2d(b, 1, 3, padding=1)
        nn.init.zeros_(self.out_c.weight); nn.init.zeros_(self.out_c.bias)

    def forward(self, x, lam, cls):
        half = torch.exp(torch.arange(32, device=x.device) * (-math.log(200) / 31))
        ang = lam * half[None, :] * 4.0                        # lam in ~[-10,10]
        t = self.temb(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)) + self.cemb(cls)
        h1 = self.d1b(self.d1a(self.cin(x), t), t)             # 32, b
        h2 = self.d2b(self.d2a(self.down1(h1), t), t)          # 16, 2b
        h3 = self.attn3(self.d3b(self.d3a(self.down2(h2), t), t))  # 8, 2b
        m = self.mid2(self.mida(self.mid1(h3, t)), t)
        u3 = self.attnu3(self.u3a(torch.cat([m, h3], 1), t))
        u2 = self.u2a(torch.cat([self.up2(u3), h2], 1), t)
        u1 = self.u1a(torch.cat([self.up1(u2), h1], 1), t)
        return self.out_c(F.silu(self.out_n(u1)))

# ------------------------------------------------------------------ schedules

def sd_linear_alphabar(T=1000, b0=0.00085, b1=0.012):
    """SD scaled-linear: betas linear in sqrt space. Non-zero terminal SNR on purpose."""
    betas = torch.linspace(b0**0.5, b1**0.5, T)**2
    ab = torch.cumprod(1 - betas, dim=0)
    return betas, ab

# ------------------------------------------------------------------ training

N_BINS, LAM_LO, LAM_HI = 24, -9.0, 10.0

def train_model(kind, steps, bs=512, b=64, lr=2e-4, seed=0, tag='mnist_fm', lum=False):
    """kind: 'fm' (u-pred, logit-normal) or 'leaky' (eps-pred, sd-linear discrete).
    lum=True: per-image random brightness bias U(0,1) — gives the dataset global-statistic
    diversity, which is the precondition for the terminal-SNR bug to bite visibly."""
    torch.manual_seed(seed)
    x_all, y_all = load_mnist()
    net = UNet(b=b).to(DEV)
    ema = UNet(b=b).to(DEV); ema.load_state_dict(net.state_dict())
    print(f'[{tag}] params: {sum(p.numel() for p in net.parameters())/1e6:.2f}M')
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    if kind == 'leaky':
        betas, ab = sd_linear_alphabar(); ab = ab.to(DEV)
        print(f'[{tag}] terminal sqrt(alphabar_T) = {ab[-1].sqrt().item():.4f}  '
              f'logSNR_T = {float(torch.log(ab[-1]/(1-ab[-1]))):.2f}')
    bin_x0 = torch.zeros(N_BINS, device=DEV); bin_cnt = torch.zeros(N_BINS, device=DEV)
    curve = []
    scaler_dtype = torch.bfloat16
    for it in range(steps):
        idx = torch.randint(0, x_all.shape[0], (bs,), device=DEV)
        x0, cls = x_all[idx], y_all[idx]
        if lum:
            beta = torch.rand(bs, 1, 1, 1, device=DEV)
            x0 = (x0 + beta).clamp(-1, 1)
        drop = torch.rand(bs, device=DEV) < 0.10
        cls = torch.where(drop, torch.full_like(cls, 10), cls)
        eps = torch.randn_like(x0)
        if kind == 'fm':
            t = torch.sigmoid(torch.randn(bs, 1, 1, 1, device=DEV))
            a, s = 1 - t, t
            x_t = a * x0 + s * eps
            target = eps - x0
        else:
            ti = torch.randint(0, 1000, (bs,), device=DEV)
            a = ab[ti].sqrt().reshape(bs, 1, 1, 1); s = (1 - ab[ti]).sqrt().reshape(bs, 1, 1, 1)
            x_t = a * x0 + s * eps
            target = eps
        lam = logsnr(a, s).reshape(bs, 1)
        with torch.autocast('cuda', dtype=scaler_dtype):
            out = net(x_t, lam, cls)
        out = out.float()
        loss = ((out - target)**2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for p_e, p in zip(ema.parameters(), net.parameters()):
                p_e.mul_(0.9995).add_(p, alpha=0.0005)
            x0h = x_t - s * out if kind == 'fm' else (x_t - s * out) / a.clamp(min=1e-4)
            per_x0 = ((x0h - x0)**2).mean(dim=(1, 2, 3))
            bins = ((lam[:, 0] - LAM_LO) / (LAM_HI - LAM_LO) * N_BINS).long().clamp(0, N_BINS - 1)
            bin_x0.mul_(0.998).index_add_(0, bins, 0.002 * per_x0.detach())
            bin_cnt.mul_(0.998).index_add_(0, bins, 0.002 * torch.ones_like(per_x0))
        if it % 250 == 0 or it == steps - 1:
            cnt = bin_cnt.cpu().numpy(); bx0 = bin_x0.cpu().numpy()
            snap = np.where(cnt > 1e-8, bx0 / np.maximum(cnt, 1e-12), np.nan)
            curve.append({'step': it, 'loss': float(loss.item()),
                          'x0_per_bin': [None if np.isnan(v) else float(v) for v in snap]})
            if it % 2000 == 0:
                print(f'[{tag}] step {it:6d} loss {loss.item():.4f}')
    torch.save({'ema': ema.state_dict(), 'kind': kind, 'b': b, 'curve': curve}, os.path.join(RUNS, f'{tag}.pt'))
    print(f'[{tag}] saved.')

def train_1step_student(steps=6000, bs=256, b=48, n_pairs=40000, teacher_tag='mnist_fm'):
    """Generate (noise, teacher-ODE-endpoint, class) pairs, regress noise->x0 in ONE step.
    The honest naive baseline: shows exactly what breaks (blur at mode boundaries)."""
    ck = torch.load(os.path.join(RUNS, f'{teacher_tag}.pt'), map_location=DEV, weights_only=False)
    teacher = UNet(b=ck['b']).to(DEV); teacher.load_state_dict(ck['ema']); teacher.eval()
    g = torch.Generator(device=DEV).manual_seed(42)
    # t grid must start BELOW 1.0: at t=1 exactly, lam = log(0) = -inf -> NaN in the
    # sinusoidal embedding -> every pair silently poisoned. (Reproduced once; see §10.)
    T_START, T_END = 0.985, 0.002
    xs, ys, x0s = [], [], []
    with torch.no_grad():
        for i in range(0, n_pairs, 500):
            n = min(500, n_pairs - i)
            x1 = torch.randn(n, 1, 32, 32, device=DEV, generator=g)
            cls = torch.randint(0, 10, (n,), device=DEV, generator=g)
            x = x1.clone()
            ts = torch.linspace(T_START, T_END, 33, device=DEV)
            for k in range(32):
                t = ts[k]; lam = logsnr(1 - t, t).reshape(1, 1).expand(n, 1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    u = teacher(x, lam, cls).float()
                x = x + (ts[k+1] - ts[k]) * u
            xs.append(x1.cpu()); ys.append(cls.cpu()); x0s.append(x.cpu())
            if i % 10000 == 0: print(f'[pairs] {i}/{n_pairs}')
    X1, CLS, X0 = torch.cat(xs), torch.cat(ys), torch.cat(x0s)
    assert torch.isfinite(X0).all(), 'teacher endpoints contain NaN/inf — check the t grid'
    print(f'[pairs] x0 std={X0.std():.3f} mean={X0.mean():.3f}')
    torch.save({'x1': X1[:64], 'cls': CLS[:64], 'x0': X0[:64]}, os.path.join(RUNS, 'distill_pairs_preview.pt'))
    student = UNet(b=b).to(DEV)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    span = T_START - T_END
    lam1 = logsnr(torch.tensor(1 - T_START), torch.tensor(T_START)).item()  # start-of-integration lambda
    for it in range(steps):
        idx = torch.randint(0, X1.shape[0], (bs,))
        x1, cls, x0 = X1[idx].to(DEV), CLS[idx].to(DEV), X0[idx].to(DEV)
        lam = torch.full((bs, 1), lam1, device=DEV)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = x1 - span * student(x1, lam, cls).float()   # x0 = x1 - span * mean-velocity
        loss = ((pred - x0)**2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if it % 1000 == 0: print(f'[student] {it} loss {loss.item():.4f}')
    torch.save({'ema': student.state_dict(), 'kind': 'fm', 'b': b}, os.path.join(RUNS, 'mnist_1step.pt'))
    print('[student] saved.')

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('fm', 'all'):
        train_model('fm', steps=22000, tag='mnist_fm')
    if which in ('leaky', 'all'):
        train_model('leaky', steps=14000, tag='mnist_leaky')
    if which in ('student', 'all'):
        train_1step_student()
    if which == 'lum':  # luminance-diverse pair for the museum headline exhibit
        train_model('fm', steps=12000, b=48, tag='mnist_lumfm', lum=True)
        train_model('leaky', steps=12000, b=48, tag='mnist_lumleaky', lum=True)
