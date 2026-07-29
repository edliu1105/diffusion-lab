// axis_refs.js — every variable on this page traces to a paper.
// Chips show ALGORITHM names (links go to the arXiv source).

export const REFS = {
  vdm:      { label: 'VDM · λ=logSNR 轴', url: 'https://arxiv.org/abs/2107.00630' },
  ddpm:     { label: 'DDPM', url: 'https://arxiv.org/abs/2006.11239' },
  iddpm:    { label: 'iDDPM · cosine 调度', url: 'https://arxiv.org/abs/2102.09672' },
  ddim:     { label: 'DDIM', url: 'https://arxiv.org/abs/2010.02502' },
  sde:      { label: 'Score SDE', url: 'https://arxiv.org/abs/2011.13456' },
  edm:      { label: 'EDM', url: 'https://arxiv.org/abs/2206.00364' },
  edm2:     { label: 'EDM2', url: 'https://arxiv.org/abs/2312.02696' },
  fm:       { label: 'Flow Matching', url: 'https://arxiv.org/abs/2210.02747' },
  rf:       { label: 'Rectified Flow / ReFlow', url: 'https://arxiv.org/abs/2209.03003' },
  sd3:      { label: 'SD3 · logit-normal + shift', url: 'https://arxiv.org/abs/2403.03206' },
  vpred:    { label: 'v-prediction / Progressive Distillation', url: 'https://arxiv.org/abs/2202.00512' },
  kingma23: { label: '损失权重统一定理', url: 'https://arxiv.org/abs/2303.00848' },
  cm:       { label: 'Consistency Models', url: 'https://arxiv.org/abs/2303.01469' },
  ctm:      { label: 'CTM · 任意区间映射', url: 'https://arxiv.org/abs/2310.02279' },
  scm:      { label: 'sCM / TrigFlow', url: 'https://arxiv.org/abs/2410.11081' },
  meanflow: { label: 'MeanFlow · 平均速度', url: 'https://arxiv.org/abs/2505.13447' },
  shortcut: { label: 'Shortcut Models', url: 'https://arxiv.org/abs/2410.12557' },
  dmd:      { label: 'DMD / DMD2', url: 'https://arxiv.org/abs/2405.14867' },
  selfflow: { label: 'Self-Flow · token 级双 λ 调度', url: 'https://arxiv.org/abs/2603.06507' },
  df:       { label: 'Diffusion Forcing', url: 'https://arxiv.org/abs/2407.01392' },
  causvid:  { label: 'CausVid', url: 'https://arxiv.org/abs/2412.07772' },
  selfforce:{ label: 'Self-Forcing', url: 'https://arxiv.org/abs/2506.08009' },
  tweedie:  { label: 'Tweedie 公式', url: 'https://efron.ckirby.su.domains/papers/2011TweediesFormula.pdf' },
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
