// p_s11.js — §11: cheat sheet, reproduction, and the JS-vs-PyTorch fixture check.

import { R } from './models.js';
import { vpFromLam, fmTFromLam, sigVEFromLam } from './nn.js';
import { fmEuler, runToEnd } from './diff.js';
import { C } from './charts.js';

export async function init(mount) {
  const lamRows = [-8, -4, -2, 0, 2, 4, 8].map(l => {
    const { a, s } = vpFromLam(l);
    return `<tr><td class="num">${l}</td><td class="num">${a.toFixed(4)}</td><td class="num">${s.toFixed(4)}</td>
      <td class="num">${fmTFromLam(l).toFixed(4)}</td><td class="num">${sigVEFromLam(l).toFixed(3)}</td></tr>`;
  }).join('');

  mount.innerHTML = `
  <h3>11.1 速查卡</h3>
  <div class="grid2">
    <div>
      <h4>五衣换算(给定 x_t 与 λ;VP 系 α²+σ²=1)</h4>
      <pre>x̂₀    = (x_t − σ·ε̂)/α        ε̂ = (x_t − α·x̂₀)/σ
v̂     = α·ε̂ − σ·x̂₀           x̂₀ = α·x_t − σ·v̂
score = −ε̂/σ = (α·x̂₀ − x_t)/σ²
û(FM) = ε̂ − x̂₀ = (x_t^{FM} − x̂₀)/t
坐标缩放:x^{FM} = x^{VP}·(1−t)/α,x^{VE} = x^{VP}/α</pre>
      <h4>loss 汇率(× ‖x̂₀−x₀‖²,fp64 已验证)</h4>
      <pre>ε: e^λ      v: 1+e^λ      x₀: 1
u(FM 直线): (1+e^{λ/2})²   ≈ v 的 1–2 倍
EDM: (σ²+σ_d²)/(σ·σ_d)²  [x₀ 空间]</pre>
    </div>
    <div>
      <h4>λ ↔ 各家时间轴</h4>
      <table class="data"><tr><th>λ</th><th>α</th><th>σ</th><th>t_FM</th><th>σ_VE</th></tr>${lamRows}</table>
      <p class="small">t_FM = sigmoid(−λ/2);σ_VE = e^{−λ/2};任何"时间"都是 λ 的单调重参数。</p>
    </div>
  </div>
  <h4>采样器一行本质</h4>
  <pre>DDIM:     x ← α'·x̂₀ + σ'·ε̂                    (跳到新 λ 的"当前最优重构")
DDPM:     同上,但 σ' 里分 τ 份额换成新噪声       (η=1 的 DDIM)
FM-Euler: x ← x + Δt·(x−x̂₀)/t                   (直线坐标里的一阶步)
Heun:     Euler 试探 + 端点斜率平均               (二阶;EDM 默认)
CFG:      x̂₀ ← x̂₀ᵘ + w·(x̂₀ᶜ − x̂₀ᵘ)           (每步的分布算术)</pre>

  <h3>11.2 复现</h3>
  <pre>python exp/common.py          # 恒等式 fp64 自检(§2/3 的换算表)
python exp/toy2d.py           # 2D 四模型 + 评测 → app/data/
python exp/distill2d.py       # reflow + 一步学生 → app/data/
python exp/mnist.py all       # MNIST 三模型 → runs/
python exp/mnist_export.py    # 全部图集 → app/data/mnist/
python exp/make_fixtures.py   # 对拍数据 → app/data/fixtures.json
python -m http.server 8737 --directory app   # 本实验室</pre>

  <h3>11.3 [证明] 浏览器 ↔ PyTorch 对拍</h3>
  <p>你在本页跑的每个"LIVE"实验,底层是 JS 重写的前向传播与采样器。它们和训练侧的 PyTorch 是否逐位一致?现场验:48 个随机 (x, λ) 点 × 每个模型的 denoiseVP,外加一整条 24 步 FM-Euler 轨迹,与 Python 预存值对比:</p>
  <div class="lab"><div class="labhead"><span class="id">证明 11.3</span><span class="t">运行时对拍</span><span class="live">LIVE</span>
  <button class="primary" id="fx-run" style="margin-left:auto">运行对拍</button></div>
  <div class="readout" id="fx-out" style="min-height:120px">fixtures.json 由 exp/make_fixtures.py 生成。</div></div>

  <h3>11.4 文件索引</h3>
  <table class="data wide">
    <tr><th>文件</th><th>内容</th></tr>
    <tr><td class="m">exp/common.py</td><td>统一数学核心:路径/换算/采样器/解析 GMM/fp64 自检——全站的"宪法"</td></tr>
    <tr><td class="m">exp/toy2d.py · distill2d.py</td><td>2D 四方言训练 + 蒸馏(reflow/平均速度学生)</td></tr>
    <tr><td class="m">exp/mnist.py · mnist_export.py</td><td>MNIST 健康/泄漏/一步学生 + 全部图集导出</td></tr>
    <tr><td class="m">app/js/nn.js · gmm.js · diff.js</td><td>浏览器运行时:MLP 前向 / 解析真值 / 采样器(与 Python 对拍)</td></tr>
    <tr><td class="m">app/js/p_*.js</td><td>各章面板(叙事+实验共居一处)</td></tr>
  </table>`;

  document.getElementById('fx-run').addEventListener('click', async () => {
    const out = document.getElementById('fx-out');
    let fx;
    try { fx = await (await fetch('data/fixtures.json')).json(); }
    catch { out.innerHTML = '<span style="color:var(--warn)">fixtures.json 未生成,先跑 exp/make_fixtures.py。</span>'; return; }
    const n = fx.lam.length;
    const xs = new Float32Array(n * 2);
    fx.x.forEach((p, i) => { xs[i * 2] = p[0]; xs[i * 2 + 1] = p[1]; });
    const lam = new Float32Array(fx.lam);
    const lines = [];
    const check = (name, got, want) => {
      let mx = 0;
      for (let i = 0; i < n; i++) {
        mx = Math.max(mx, Math.abs(got[i * 2] - want[i][0]), Math.abs(got[i * 2 + 1] - want[i][1]));
      }
      const ok = mx < 2e-3;
      lines.push(`${name.padEnd(12)} max|Δ| = <b style="color:${ok ? C.ok : C.bad}">${mx.toExponential(2)}</b> ${ok ? 'PASS' : 'FAIL'}`);
    };
    check('analytic', R.gmm.denoiseVP(xs, lam, n), fx.analytic);
    for (const [name, want] of Object.entries(fx.models)) {
      if (!R.models[name]) { lines.push(`${name}: 模型未加载`); continue; }
      check(name, R.models[name].denoiseVP(xs, lam, n), want);
    }
    if (fx.fme_run && R.models.u_fm) {
      const m = fx.fme_run.x1.length;
      const x1 = new Float32Array(m * 2);
      fx.fme_run.x1.forEach((p, i) => { x1[i * 2] = p[0]; x1[i * 2 + 1] = p[1]; });
      const s = fmEuler((x, l, k) => R.models.u_fm.denoiseVP(x, l, k), x1, m, fx.fme_run.steps,
        { tStart: fx.fme_run.t_start, tEnd: fx.fme_run.t_end });
      runToEnd(s);
      let mx = 0;
      for (let i = 0; i < m; i++) mx = Math.max(mx, Math.abs(s.xs[i * 2] - fx.fme_run.end[i][0]), Math.abs(s.xs[i * 2 + 1] - fx.fme_run.end[i][1]));
      const ok = mx < 5e-3;
      lines.push(`fme 24步全程   max|Δ| = <b style="color:${ok ? C.ok : C.bad}">${mx.toExponential(2)}</b> ${ok ? 'PASS' : 'FAIL'}`);
    }
    lines.push('', '<span class="dim">差异来源仅为 fp32 累加顺序;阈值 2e-3(24 步轨迹 5e-3)。</span>');
    out.innerHTML = lines.join('\n');
  });
}
