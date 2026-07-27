"""Bonus suite: class-conditional FM on CIFAR-10 (color -> CFG saturation & texture demos).
Same recipe as mnist.py, 3 channels, slightly bigger UNet. Non-blocking for delivery.
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from mnist import ResBlock, Attn, RUNS
from common import logsnr

DEV = 'cuda'

class UNet3(nn.Module):
    def __init__(self, b=96, tdim=384, n_class=11):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.cemb = nn.Embedding(n_class, tdim)
        self.cin = nn.Conv2d(3, b, 3, padding=1)
        self.d1a, self.d1b = ResBlock(b, b, tdim), ResBlock(b, b, tdim)
        self.down1 = nn.Conv2d(b, b * 2, 3, stride=2, padding=1)
        self.d2a, self.d2b = ResBlock(b * 2, b * 2, tdim), ResBlock(b * 2, b * 2, tdim)
        self.attn2 = Attn(b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 2, 3, stride=2, padding=1)
        self.d3a, self.d3b = ResBlock(b * 2, b * 2, tdim), ResBlock(b * 2, b * 2, tdim)
        self.attn3 = Attn(b * 2)
        self.mid1, self.mida, self.mid2 = ResBlock(b * 2, b * 2, tdim), Attn(b * 2), ResBlock(b * 2, b * 2, tdim)
        self.u3a = ResBlock(b * 4, b * 2, tdim); self.attnu3 = Attn(b * 2)
        self.up2 = nn.ConvTranspose2d(b * 2, b * 2, 4, stride=2, padding=1)
        self.u2a = ResBlock(b * 4, b * 2, tdim); self.attnu2 = Attn(b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.u1a = ResBlock(b * 2, b, tdim)
        self.out_n = nn.GroupNorm(8, b); self.out_c = nn.Conv2d(b, 3, 3, padding=1)
        nn.init.zeros_(self.out_c.weight); nn.init.zeros_(self.out_c.bias)
    def forward(self, x, lam, cls):
        import math
        half = torch.exp(torch.arange(32, device=x.device) * (-math.log(200) / 31))
        ang = lam * half[None, :] * 4.0
        t = self.temb(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)) + self.cemb(cls)
        h1 = self.d1b(self.d1a(self.cin(x), t), t)
        h2 = self.attn2(self.d2b(self.d2a(self.down1(h1), t), t))
        h3 = self.attn3(self.d3b(self.d3a(self.down2(h2), t), t))
        m = self.mid2(self.mida(self.mid1(h3, t)), t)
        u3 = self.attnu3(self.u3a(torch.cat([m, h3], 1), t))
        u2 = self.attnu2(self.u2a(torch.cat([self.up2(u3), h2], 1), t))
        u1 = self.u1a(torch.cat([self.up1(u2), h1], 1), t)
        return self.out_c(F.silu(self.out_n(u1)))

def main(steps=46000, bs=384, lr=2e-4):
    from torchvision import datasets
    ds = datasets.CIFAR10(os.path.join(RUNS, 'data'), train=True, download=True)
    x_all = (torch.tensor(ds.data).float() / 255 * 2 - 1).permute(0, 3, 1, 2).contiguous().to(DEV)
    y_all = torch.tensor(ds.targets).to(DEV)
    torch.manual_seed(0)
    net = UNet3().to(DEV)
    ema = UNet3().to(DEV); ema.load_state_dict(net.state_dict())
    print(f'[cifar] params {sum(p.numel() for p in net.parameters())/1e6:.1f}M')
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    N_BINS = 24
    bin_x0 = torch.zeros(N_BINS, device=DEV); bin_cnt = torch.zeros(N_BINS, device=DEV)
    curve = []
    for it in range(steps):
        idx = torch.randint(0, x_all.shape[0], (bs,), device=DEV)
        x0, cls = x_all[idx], y_all[idx]
        cls = torch.where(torch.rand(bs, device=DEV) < 0.10, torch.full_like(cls, 10), cls)
        eps = torch.randn_like(x0)
        t = torch.sigmoid(torch.randn(bs, 1, 1, 1, device=DEV))
        x_t = (1 - t) * x0 + t * eps
        lam = logsnr(1 - t, t).reshape(bs, 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = net(x_t, lam, cls)
        loss = ((out.float() - (eps - x0)) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.9997).add_(p, alpha=0.0003)
            x0h = x_t - t * out.float()
            per = ((x0h - x0) ** 2).mean(dim=(1, 2, 3))
            bins = ((lam[:, 0] + 9) / 19 * N_BINS).long().clamp(0, N_BINS - 1)
            bin_x0.mul_(0.998).index_add_(0, bins, 0.002 * per.detach())
            bin_cnt.mul_(0.998).index_add_(0, bins, 0.002 * torch.ones_like(per))
        if it % 500 == 0:
            cnt = bin_cnt.cpu().numpy(); b0 = bin_x0.cpu().numpy()
            snap = np.where(cnt > 1e-8, b0 / np.maximum(cnt, 1e-12), np.nan)
            curve.append({'step': it, 'loss': float(loss.item()),
                          'x0_per_bin': [None if np.isnan(v) else float(v) for v in snap]})
            if it % 4000 == 0:
                print(f'[cifar] {it} loss {loss.item():.4f}')
        if it % 10000 == 9999:
            torch.save({'ema': ema.state_dict(), 'b': 96, 'curve': curve, 'kind': 'fm'},
                       os.path.join(RUNS, 'cifar_fm.pt'))
    torch.save({'ema': ema.state_dict(), 'b': 96, 'curve': curve, 'kind': 'fm'}, os.path.join(RUNS, 'cifar_fm.pt'))
    print('[cifar] done.')

if __name__ == '__main__':
    main()
