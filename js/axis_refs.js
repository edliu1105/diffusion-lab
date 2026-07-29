// axis_refs.js — every variable on this page traces to a paper. One registry, one chip style.

export const REFS = {
  vdm:      { label: 'Kingma et al. 2021 · VDM(λ=logSNR 轴)', url: 'https://arxiv.org/abs/2107.00630' },
  ddpm:     { label: 'Ho et al. 2020 · DDPM', url: 'https://arxiv.org/abs/2006.11239' },
  iddpm:    { label: 'Nichol & Dhariwal 2021 · iDDPM(cosine)', url: 'https://arxiv.org/abs/2102.09672' },
  ddim:     { label: 'Song et al. 2020 · DDIM(η)', url: 'https://arxiv.org/abs/2010.02502' },
  sde:      { label: 'Song et al. 2021 · Score SDE', url: 'https://arxiv.org/abs/2011.13456' },
  edm:      { label: 'Karras et al. 2022 · EDM(σ 网格/churn/预条件)', url: 'https://arxiv.org/abs/2206.00364' },
  edm2:     { label: 'Karras et al. 2024 · EDM2', url: 'https://arxiv.org/abs/2312.02696' },
  fm:       { label: 'Lipman et al. 2022 · Flow Matching', url: 'https://arxiv.org/abs/2210.02747' },
  rf:       { label: 'Liu et al. 2022 · Rectified Flow / ReFlow', url: 'https://arxiv.org/abs/2209.03003' },
  sd3:      { label: 'Esser et al. 2024 · SD3(logit-normal + shift)', url: 'https://arxiv.org/abs/2403.03206' },
  vpred:    { label: 'Salimans & Ho 2022 · v-prediction / PD', url: 'https://arxiv.org/abs/2202.00512' },
  kingma23: { label: 'Kingma & Gao 2023 · 权重等价定理', url: 'https://arxiv.org/abs/2303.00848' },
  cm:       { label: 'Song et al. 2023 · Consistency Models', url: 'https://arxiv.org/abs/2303.01469' },
  ctm:      { label: 'Kim et al. 2023 · CTM(任意区间)', url: 'https://arxiv.org/abs/2310.02279' },
  scm:      { label: 'Lu & Song 2024 · sCM / TrigFlow', url: 'https://arxiv.org/abs/2410.11081' },
  meanflow: { label: 'Geng et al. 2025 · MeanFlow(平均速度)', url: 'https://arxiv.org/abs/2505.13447' },
  shortcut: { label: 'Frans et al. 2024 · Shortcut Models', url: 'https://arxiv.org/abs/2410.12557' },
  dmd:      { label: 'Yin et al. 2023/24 · DMD / DMD2', url: 'https://arxiv.org/abs/2405.14867' },
  df:       { label: 'Chen et al. 2024 · Diffusion Forcing(逐帧独立噪声)', url: 'https://arxiv.org/abs/2407.01392' },
  causvid:  { label: 'Yin et al. 2024 · CausVid(双向→因果蒸馏)', url: 'https://arxiv.org/abs/2412.07772' },
  selfforce:{ label: 'Huang et al. 2025 · Self-Forcing(自回滚训练)', url: 'https://arxiv.org/abs/2506.08009' },
  tweedie:  { label: 'Efron 2011 · Tweedie 公式(E[x₀|x_t] ↔ score)', url: 'https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf' },
};

export function refChip(...keys) {
  return keys.map(k => {
    const r = REFS[k];
    return `<a class="ref" href="${r.url}" target="_blank" rel="noopener">${r.label}</a>`;
  }).join('');
}

export function refList() {
  return Object.values(REFS).map(r => `<a class="ref" href="${r.url}" target="_blank" rel="noopener">${r.label}</a>`).join('');
}
