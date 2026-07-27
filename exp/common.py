"""Unified diffusion/flow math core.

Single frame everything lives in:
    x_t = alpha(t) * x0 + sigma(t) * eps,   eps ~ N(0, I)
    lambda := logSNR = 2*(log alpha - log sigma)   (universal noise axis)

Coordinate systems (choices of alpha/sigma over a scalar "time"):
    VP  (DDPM):  alpha^2 + sigma^2 = 1          (cosine schedule here)
    VE  (EDM) :  alpha = 1, sigma in [smin,smax]
    FM  linear:  alpha = 1-t, sigma = t, t in [0,1]

Prediction heads are affine re-labelings of one object D(x,lam) := E[x0|x_t]:
    eps_hat  = (x_t - alpha*x0_hat) / sigma
    v_hat    = alpha*eps_hat - sigma*x0_hat          (VP convention)
    score    = -eps_hat / sigma                      (= grad log p_t)
    u_hat    = eps_hat - x0_hat                      (FM linear-path velocity)

All losses are weighted x0-losses (verified numerically in selftest):
    ||eps_hat-eps||^2 = e^lam                * ||x0_hat-x0||^2
    ||v_hat  -v  ||^2 = (1+e^lam)            * ||x0_hat-x0||^2   (VP)
    ||u_hat  -u  ||^2 = (1+e^(lam/2))^2      * ||x0_hat-x0||^2   (FM linear)
"""
import math
import numpy as np
import torch

# ----------------------------------------------------------------------------
# coordinate systems: map scalar time <-> (alpha, sigma) <-> logSNR
# ----------------------------------------------------------------------------

def vp_cosine_alpha_sigma(t, s=0.008):
    """DDPM cosine schedule, continuous t in [0,1]. Returns (alpha, sigma)."""
    f = torch.cos((t + s) / (1 + s) * math.pi / 2).clamp(min=1e-5)
    f0 = math.cos(s / (1 + s) * math.pi / 2)
    alpha = (f / f0).clamp(1e-5, 1.0)
    sigma = (1 - alpha**2).clamp(min=1e-10).sqrt()
    return alpha, sigma

def fm_alpha_sigma(t):
    """Rectified-flow / FM linear path. t=0 data, t=1 noise."""
    return 1 - t, t

def ve_alpha_sigma(sigma):
    return torch.ones_like(sigma), sigma

def logsnr(alpha, sigma):
    return 2 * (torch.log(alpha) - torch.log(sigma))

def fm_t_from_logsnr(lam):
    """FM linear: (1-t)/t = e^(lam/2)  =>  t = sigmoid(-lam/2)."""
    return torch.sigmoid(-lam / 2)

def vp_alpha_sigma_from_logsnr(lam):
    """VP: alpha^2 = sigmoid(lam), sigma^2 = sigmoid(-lam)."""
    return torch.sigmoid(lam).sqrt(), torch.sigmoid(-lam).sqrt()

# ----------------------------------------------------------------------------
# prediction-head conversions (the "five dresses"). All exact, all affine.
# Given x_t and any one head, recover any other.
# ----------------------------------------------------------------------------

def x0_from_eps(x_t, eps, alpha, sigma):   return (x_t - sigma * eps) / alpha
def eps_from_x0(x_t, x0, alpha, sigma):    return (x_t - alpha * x0) / sigma
def v_from_x0_eps(x0, eps, alpha, sigma):  return alpha * eps - sigma * x0
def x0_from_v(x_t, v, alpha, sigma):       return alpha * x_t - sigma * v      # needs a^2+s^2=1
def eps_from_v(x_t, v, alpha, sigma):      return sigma * x_t + alpha * v      # needs a^2+s^2=1
def score_from_eps(eps, sigma):            return -eps / sigma
def eps_from_score(score, sigma):          return -sigma * score
def u_from_x0_eps(x0, eps):                return eps - x0                     # FM linear path
def x0_from_u(x_t, u, t):                  return x_t - t * u                  # FM linear path
def eps_from_u(x_t, u, t):                 return x_t + (1 - t) * u            # FM linear path

# effective weights turning each native loss into  w(lam) * ||x0_hat - x0||^2
def w_eps(lam):  return torch.exp(lam)
def w_v(lam):    return 1 + torch.exp(lam)
def w_u(lam):    return (1 + torch.exp(lam / 2))**2
def w_x0(lam):   return torch.ones_like(lam)
def w_edm(lam, sigma_data):
    """EDM lambda(sigma) weight, expressed as an x0-loss weight.
    EDM loss = lam_w * ||D - x0||^2 with lam_w = (s^2+sd^2)/(s*sd)^2, alpha=1 => e^-lam = s^2."""
    s2 = torch.exp(-lam)
    return (s2 + sigma_data**2) / (s2 * sigma_data**2)

# ----------------------------------------------------------------------------
# analytic ground truth: Gaussian mixture data => closed-form E[x0|x_t]
# ----------------------------------------------------------------------------

class GMM:
    """Isotropic Gaussian mixture. Everything closed-form under the forward process.

    p_t(x) = sum_i w_i N(x; alpha*mu_i, (alpha^2 s_i^2 + sigma^2) I)
    E[x0|x_t] = sum_i r_i(x) * [mu_i + (alpha s_i^2 / (alpha^2 s_i^2 + sigma^2)) (x - alpha mu_i)]
      with responsibilities r_i from the noisy mixture.
    """
    def __init__(self, mus, stds, weights=None, device='cpu'):
        self.mu = torch.as_tensor(mus, dtype=torch.float32, device=device)      # (K, d)
        self.std = torch.as_tensor(stds, dtype=torch.float32, device=device)    # (K,)
        K = self.mu.shape[0]
        w = torch.ones(K, device=device) if weights is None else torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.w = w / w.sum()

    def sample(self, n, generator=None):
        idx = torch.multinomial(self.w, n, replacement=True, generator=generator)
        x = self.mu[idx] + self.std[idx, None] * torch.randn(n, self.mu.shape[1], device=self.mu.device, generator=generator)
        return x, idx

    def _posterior(self, x_t, alpha, sigma):
        """Shared shapes: a,s -> (B,1); returns r (B,K), post_mean (B,K,d), post_var (B,K per-dim)."""
        B = x_t.shape[0]
        a = alpha.reshape(-1, 1).expand(B, 1)                                    # (B,1)
        s = sigma.reshape(-1, 1).expand(B, 1)                                    # (B,1)
        var_t = (a * self.std[None, :])**2 + s**2                                # (B,K)
        diff = x_t[:, None, :] - a[:, :, None] * self.mu[None, :, :]             # (B,K,d)
        logp = -0.5 * (diff**2).sum(-1) / var_t - 0.5 * x_t.shape[-1] * torch.log(var_t) + torch.log(self.w)[None, :]
        r = torch.softmax(logp, dim=1)                                           # (B,K)
        gain = a * self.std[None, :]**2 / var_t                                  # (B,K)
        post_mean = a[:, :, None] * 0 + self.mu[None, :, :] + gain[:, :, None] * diff   # (B,K,d)
        post_var = self.std[None, :]**2 * s**2 / var_t                           # (B,K) per-dim
        return r, post_mean, post_var

    def denoise(self, x_t, alpha, sigma, only_component=None):
        """E[x0 | x_t] under x_t = alpha x0 + sigma eps."""
        r, post_mean, _ = self._posterior(x_t, alpha, sigma)
        if only_component is not None:  # conditional-on-class denoiser (for CFG analytics)
            r = torch.zeros_like(r); r[:, only_component] = 1.0
        return (r[..., None] * post_mean).sum(1)

    def x0_posterior_var(self, x_t, alpha, sigma):
        """E[||x0 - E[x0|x_t]||^2 | x_t]: the irreducible loss floor, pointwise."""
        r, post_mean, post_var = self._posterior(x_t, alpha, sigma)
        mix_mean = (r[..., None] * post_mean).sum(1)                             # (B,d)
        d = x_t.shape[-1]
        second = (r * (post_var * d + (post_mean**2).sum(-1))).sum(1)
        return second - (mix_mean**2).sum(-1)

def gmm_ring(K=6, radius=1.5, std=0.22, device='cpu'):
    ang = torch.arange(K) * (2 * math.pi / K)
    mus = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=1)
    return GMM(mus, torch.full((K,), std), device=device)

# ----------------------------------------------------------------------------
# samplers: every one consumes the SAME callable  denoise(x, lam) -> x0_hat
# (that code shape is the thesis: one function, many samplers)
# ----------------------------------------------------------------------------

@torch.no_grad()
def sample_fm_euler(denoise, x1, steps, t_start=1.0, t_end=0.0, record=False):
    """Euler on dx/dt = u = (x - x0_hat)/t along the FM linear path."""
    x = x1.clone(); traj = [x.clone()]
    ts = torch.linspace(t_start, t_end, steps + 1, device=x.device)
    for i in range(steps):
        t = ts[i].expand(x.shape[0], 1)
        lam = logsnr(1 - t, t.clamp(min=1e-6))
        u = (x - denoise(x, lam)) / t.clamp(min=1e-6)
        x = x + (ts[i+1] - ts[i]) * u
        if record: traj.append(x.clone())
    return (x, traj) if record else x

@torch.no_grad()
def sample_edm_heun(denoise, x_sig_max, sigmas, churn=0.0, record=False, generator=None):
    """Karras Heun (2nd order) on dx/dsigma = (x - x0_hat)/sigma, VE frame.
    churn>0 re-injects noise each step (S_churn-style) => stochastic sampler."""
    x = x_sig_max.clone(); traj = [x.clone()]
    n = len(sigmas) - 1
    for i in range(n):
        s, s_next = sigmas[i], sigmas[i+1]
        if churn > 0 and 0 < s:
            gamma = min(churn / n, math.sqrt(2) - 1)
            s_hat = s * (1 + gamma)
            x = x + math.sqrt(max(s_hat**2 - s**2, 0.0)) * torch.randn(x.shape, device=x.device, generator=generator)
            s = s_hat
        lam = logsnr(torch.ones(1, device=x.device), torch.full((1,), s, device=x.device)).expand(x.shape[0], 1)
        d = (x - denoise(x, lam)) / s
        x_next = x + (s_next - s) * d
        if s_next > 0:  # Heun correction
            lam2 = logsnr(torch.ones(1, device=x.device), torch.full((1,), s_next, device=x.device)).expand(x.shape[0], 1)
            d2 = (x_next - denoise(x_next, lam2)) / s_next
            x_next = x + (s_next - s) * 0.5 * (d + d2)
        x = x_next
        if record: traj.append(x.clone())
    return (x, traj) if record else x

def karras_sigmas(n, smin=0.02, smax=8.0, rho=7.0):
    i = np.arange(n + 1) / n
    s = (smax**(1/rho) + i * (smin**(1/rho) - smax**(1/rho)))**rho
    s[-1] = 0.0
    return s

@torch.no_grad()
def sample_ddim(denoise, x_T, lams, eta=0.0, record=False, generator=None):
    """DDIM in VP frame over a logSNR grid (increasing lam). eta=1 ~ ancestral DDPM.
    Step: x_{s} = alpha_s x0_hat + sqrt(sigma_s^2 - tau^2) eps_hat + tau z."""
    x = x_T.clone(); traj = [x.clone()]
    for i in range(len(lams) - 1):
        lam, lam_s = lams[i], lams[i+1]
        a, s = vp_alpha_sigma_from_logsnr(lam)
        a_s, s_s = vp_alpha_sigma_from_logsnr(lam_s)
        x0h = denoise(x, lam.expand(x.shape[0], 1))
        epsh = (x - a * x0h) / s
        # DDIM(eta): tau interpolates deterministic (0) <-> ancestral (1)
        tau = eta * s_s / s * (1 - (a / a_s)**2).clamp(min=0).sqrt()
        tau = torch.minimum(tau, s_s)
        x = a_s * x0h + (s_s**2 - tau**2).clamp(min=0).sqrt() * epsh
        if tau > 0:
            x = x + tau * torch.randn(x.shape, device=x.device, generator=generator)
        if record: traj.append(x.clone())
    return (x, traj) if record else x

# ----------------------------------------------------------------------------
# self-tests: every identity claimed above, verified numerically
# ----------------------------------------------------------------------------

def selftest(verbose=True):
    torch.manual_seed(0)
    report = {}
    B, d = 4096, 2
    # fp64: these are exact identities; float32 cancellation would pollute the check
    x0 = (torch.randn(B, d) * 1.3 + 0.2).double()
    eps = torch.randn(B, d).double()

    # --- head round-trips at random logSNR, VP frame
    lam = (torch.rand(B, 1) * 16 - 8).double()
    a, s = vp_alpha_sigma_from_logsnr(lam)
    x_t = a * x0 + s * eps
    v = v_from_x0_eps(x0, eps, a, s)
    checks = {
        'x0_from_eps': (x0_from_eps(x_t, eps, a, s) - x0),
        'eps_from_x0': (eps_from_x0(x_t, x0, a, s) - eps),
        'x0_from_v':   (x0_from_v(x_t, v, a, s) - x0),
        'eps_from_v':  (eps_from_v(x_t, v, a, s) - eps),
        'eps_from_score': (eps_from_score(score_from_eps(eps, s), s) - eps),
    }
    # FM frame round-trip
    t = (torch.rand(B, 1) * 0.98 + 0.01).double()
    x_t_fm = (1 - t) * x0 + t * eps
    u = u_from_x0_eps(x0, eps)
    checks['x0_from_u'] = x0_from_u(x_t_fm, u, t) - x0
    checks['eps_from_u'] = eps_from_u(x_t_fm, u, t) - eps
    for k, e in checks.items():
        report[k] = e.abs().max().item()
        assert report[k] < 1e-4, (k, report[k])

    # --- loss-weight identities: perturb x0_hat, compare native losses
    x0h = x0 + 0.1 * torch.randn(B, d)
    dx2 = ((x0h - x0)**2).sum(-1)
    epsh = eps_from_x0(x_t, x0h, a, s)
    vh = a * epsh - s * x0h
    r_eps = (((epsh - eps)**2).sum(-1) / (dx2 * torch.exp(lam[:, 0]))).log().abs().max().item()
    r_v   = (((vh - v)**2).sum(-1) / (dx2 * (1 + torch.exp(lam[:, 0])))).log().abs().max().item()
    lam_fm = logsnr(1 - t, t)
    epsh_fm = eps_from_x0(x_t_fm, x0h, 1 - t, t)
    uh = epsh_fm - x0h
    r_u = (((uh - u)**2).sum(-1) / (dx2 * (1 + torch.exp(lam_fm[:, 0] / 2))**2)).log().abs().max().item()
    report['w_eps=e^lam'], report['w_v=1+e^lam'], report['w_u=(1+e^(lam/2))^2'] = r_eps, r_v, r_u
    assert max(r_eps, r_v, r_u) < 1e-3

    # --- analytic GMM: samplers agree with each other & with the data law
    gmm = gmm_ring()
    def denoise(x, lam):
        a, s = vp_alpha_sigma_from_logsnr(lam[:, :1])
        # convert VP point to VE frame: x_ve = x_vp / a, sigma_ve = s/a — or evaluate directly:
        return gmm.denoise(x, a[:, 0:1], s[:, 0:1])
    g = torch.Generator().manual_seed(7)
    xT = torch.randn(3000, 2, generator=g)

    lams = torch.linspace(-8.5, 9.0, 400).flip(0)  # high noise -> low noise? lam increasing = less noise
    lams = torch.linspace(-8.5, 9.0, 400)          # start at low lam (max noise)
    out_ddim = sample_ddim(denoise, xT, list(lams))
    # FM sampler on the same object (frame change: VP<->FM via logSNR)
    def denoise_fm(x, lam):
        tt = fm_t_from_logsnr(lam[:, :1])
        return gmm.denoise(x, 1 - tt, tt)
    out_fm = sample_fm_euler(denoise_fm, xT, steps=400)
    # EDM Heun, VE frame
    def denoise_ve(x, lam):
        s = torch.exp(-lam[:, :1] / 2)
        return gmm.denoise(x, torch.ones_like(s), s)
    out_edm = sample_edm_heun(denoise_ve, xT * 8.0, karras_sigmas(80, 0.005, 8.0))

    xs, _ = gmm.sample(3000, generator=torch.Generator().manual_seed(1))
    def mstats(x): return torch.tensor([x.mean(0)[0], x.mean(0)[1], x.std(0)[0], x.std(0)[1], (x.norm(dim=1)).mean()])
    for name, out in [('ddim_vs_data', out_ddim), ('fm_vs_data', out_fm), ('edm_vs_data', out_edm)]:
        dstat = (mstats(out) - mstats(xs)).abs().max().item()
        report[name + '_statgap'] = dstat
        assert dstat < 0.12, (name, dstat)

    # --- DDIM (many steps) == FM Euler (many steps) modulo frame change, per-sample
    # map VP trajectory endpoint to FM: they should produce near-identical samples from same xT
    gap = (out_ddim - out_fm).norm(dim=1).median().item()
    report['ddim_equals_fmEuler_median_gap'] = gap
    assert gap < 0.05, gap

    # --- loss floor: analytic conditional variance matches MC estimate at a few lams
    for L in [-4.0, 0.0, 4.0]:
        lamt = torch.full((20000, 1), L)
        a2, s2 = vp_alpha_sigma_from_logsnr(lamt)
        x0s, _ = gmm.sample(20000, generator=torch.Generator().manual_seed(2))
        xts = a2 * x0s + s2 * torch.randn(20000, 2, generator=torch.Generator().manual_seed(3))
        mc = ((x0s - gmm.denoise(xts, a2, s2))**2).sum(-1).mean().item()
        an = gmm.x0_posterior_var(xts, a2, s2).mean().item()
        report[f'floor@lam={L}_mc'] = mc; report[f'floor@lam={L}_analytic'] = an
        assert abs(mc - an) / max(an, 1e-6) < 0.05

    if verbose:
        for k, v in report.items(): print(f'  {k:36s} {v:.6g}')
    return report

if __name__ == '__main__':
    print('running selftest...')
    selftest()
    print('ALL IDENTITIES VERIFIED.')
