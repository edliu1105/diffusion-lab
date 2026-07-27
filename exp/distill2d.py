"""Distillation on the 2D world, teacher = the exported u_fm JSON (same weights the browser runs).

Three artifacts:
  u_reflow : ReFlow student. Same objective family as teacher (instantaneous velocity),
             but trained on PAIRED (x0_end, x1) couplings from the teacher ODE
             => the marginal path straightens; few-step Euler stops cutting corners.
  u_1step  : mean-velocity student. Predicts uu(x1) = (x1 - x0_end)/(t1 - t0):
             ONE Euler step of size (t1-t0) lands exactly on the teacher endpoint
             (if learned perfectly). This is the (r,t)=(0,1) special case of
             MeanFlow / the boundary case of consistency models.
  pairs    : a scatter of teacher couplings for the app.

Everything exported to app/data (same format), so the browser can run students live.
"""
import json, math, os, sys, base64
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from toy2d import MLP, LAM_SCALE, FREQS, b64f32, export_model, OUT

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T_START, T_END = 0.985, 0.002

def load_json_model(path):
    spec = json.load(open(path))
    net = MLP(n_class=0).to(DEV)
    lin = [m for m in net.net if isinstance(m, nn.Linear)]
    for m, l in zip(lin, spec['layers']):
        w = np.frombuffer(base64.b64decode(l['w']), dtype='<f4').reshape(l['shape']).copy()
        b = np.frombuffer(base64.b64decode(l['b']), dtype='<f4').copy()
        m.weight.data = torch.tensor(w, device=DEV); m.bias.data = torch.tensor(b, device=DEV)
    net.eval()
    return net

def lam_of_t(t): return 2 * torch.log((1 - t) / t)

@torch.no_grad()
def teacher_endpoints(net, x1, steps=128):
    """FM Euler from T_START to T_END, matching the browser's grid exactly."""
    x = x1.clone()
    ts = torch.linspace(T_START, T_END, steps + 1, device=DEV)
    for k in range(steps):
        t = ts[k].expand(x.shape[0], 1)
        u = net(x, lam_of_t(t))
        x = x + (ts[k + 1] - ts[k]) * u
    return x

def train_student(name, data_iter, steps=20000, lr=1e-3, init_from=None):
    torch.manual_seed(7)
    net = MLP(n_class=0).to(DEV)
    if init_from is not None:
        net.load_state_dict(init_from.state_dict())
    ema = MLP(n_class=0).to(DEV); ema.load_state_dict(net.state_dict())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    for it in range(steps):
        x_in, lam, target = data_iter()
        loss = ((net(x_in, lam) - target) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.9995).add_(p, alpha=0.0005)
        if it % 4000 == 0: print(f'  [{name}] {it} loss {loss.item():.5f}')
    return ema

def main():
    teacher = load_json_model(os.path.join(OUT, 'model_u_fm.json'))
    print('generating teacher couplings ...')
    g = torch.Generator(device=DEV).manual_seed(123)
    N = 400_000
    x1 = torch.randn(N, 2, device=DEV, generator=g)
    x0 = torch.empty_like(x1)
    for i in range(0, N, 50_000):
        x0[i:i + 50_000] = teacher_endpoints(teacher, x1[i:i + 50_000])
        print(f'  pairs {i + 50_000}/{N}')

    # --- ReFlow: instantaneous velocity on straight PAIRED couplings, t ~ U
    def reflow_batch(bs=2048):
        idx = torch.randint(0, N, (bs,), device=DEV)
        a, b = x0[idx], x1[idx]
        t = torch.rand(bs, 1, device=DEV) * (T_START - T_END) + T_END
        x_t = (1 - t) * a + t * b
        return x_t, lam_of_t(t), b - a
    print('training u_reflow ...')
    reflow = train_student('reflow', reflow_batch, init_from=teacher)
    export_model(reflow, 'u', os.path.join(OUT, 'model_u_reflow.json'))

    # --- mean-velocity 1-step: input x1 at t=T_START, predict (x1-x0)/(T_START-T_END)
    span = T_START - T_END
    def onestep_batch(bs=2048):
        idx = torch.randint(0, N, (bs,), device=DEV)
        a, b = x0[idx], x1[idx]
        lam = lam_of_t(torch.full((bs, 1), T_START, device=DEV))
        return b, lam, (b - a) / span
    print('training u_1step (mean velocity) ...')
    onestep = train_student('1step', onestep_batch)
    export_model(onestep, 'u', os.path.join(OUT, 'model_u_1step.json'),
                 extra={'mean_velocity': True, 't_start': T_START, 't_end': T_END})

    # --- pairs scatter + straightness report for the app
    with torch.no_grad():
        # straightness: E || u(x_t,t) - (x1-x0) ||^2 along own path (teacher vs reflow)
        rep = {}
        for name, net in [('teacher', teacher), ('reflow', reflow)]:
            idx = torch.randint(0, N, (8192,), device=DEV, generator=g)
            b = x1[idx]
            # straightness, operational definition: few-step endpoint drift vs own 128-step
            e128 = teacher_endpoints(net, b, steps=128)
            e2 = teacher_endpoints(net, b, steps=2)
            e8 = teacher_endpoints(net, b, steps=8)
            rep[name] = {
                'gap_2step': float((e2 - e128).norm(dim=1).mean()),
                'gap_8step': float((e8 - e128).norm(dim=1).mean()),
            }
        print('straightness (few-step endpoint drift vs own 128-step):', rep)

    pk = torch.randperm(N)[:500]
    out = {
        'pairs_x1': x1[pk].cpu().numpy().round(4).tolist(),
        'pairs_x0': x0[pk].cpu().numpy().round(4).tolist(),
        'straightness': rep,
        't_start': T_START, 't_end': T_END,
    }
    with open(os.path.join(OUT, 'distill2d_results.json'), 'w') as f:
        json.dump(out, f)
    print('wrote distill2d_results.json')

if __name__ == '__main__':
    main()
