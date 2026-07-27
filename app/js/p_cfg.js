// p_cfg.js — §6: classifier-free guidance as distribution arithmetic.
// Punchline experiment: guidance distorts EVEN WITH A PERFECT MODEL — we have the
// exact conditional & unconditional denoisers of the GMM, so this is provable here.

import { R } from './models.js';
import { Plane, MODE_COLORS } from './viz2d.js';
import { C, Chart, legendHTML } from './charts.js';
import { mulberry32, gauss } from './gmm.js';
import { fmEuler, runToEnd, fmToVP } from './diff.js';
import { vpFromLam, fmTFromLam } from './nn.js';

export async function init(mount) {
  mount.innerHTML = `
  <p>CFG 的全部实现:训练时以 10% 概率把条件替换成"空"(所以一张网同时是条件模型和无条件模型);推理时每步做一次算术:</p>
  <p style="text-align:center"><code style="font-size:15px">D_guided = D_uncond + w · (D_cond − D_uncond)</code></p>
  <p>机制层用贝叶斯翻译一次就懂了:score 版本等价于 <code>∇log p_t(x) + w·∇log p_t(c|x)</code>——
  <strong>在原分布的坡上,额外沿"分类器更确信是 c"的方向上坡</strong>。w=1 是诚实的条件模型;w&gt;1 开始采一个不存在的分布。
  关键的几何直觉(多数教程讲错或不讲):<b>p(c|x) 的坡在越过该类数据之后并不停</b>——离竞争类越远,分类器越自信。
  所以引导不是把样本"压向类中心",而是把它<b>推过中心、推出数据流形</b>,往"比真数据更像 c"的方向去。
  图像空间里,"比真红车更像红车"的方向就是更饱和、更对比、更标准姿态——这就是 CFG 过曝脸的第一性解释。
  另一个独立的后果:中间步的组合场不是任何合法扩散边缘的 score,轨迹先出流形再被拉回(Karras 组 2024 autoguidance 论文的批评)。<span class="tag f">事实</span></p>
  <p><strong>下面的实验特意用解析真值做</strong>:条件/无条件去噪器都是精确闭式解,零模型误差。测出的任何形变都是 CFG 本身的性质,不是网络的锅。</p>

  <div class="lab">
    <div class="labhead"><span class="id">实验 6.1</span><span class="t">引导强度 × 分布形变</span><span class="live">LIVE</span></div>
    <div class="ctl">
      <div class="grp"><label>场</label><select id="cf-model"><option value="analytic">解析真值(完美模型)</option><option value="u_fm_cond">u_fm_cond(训练网络)</option></select></div>
      <div class="grp"><label>目标类</label><select id="cf-cls">${[0, 1, 2, 3, 4, 5].map(k => `<option value="${k}">mode ${k}</option>`).join('')}</select></div>
      <div class="grp"><label>w</label><input type="range" id="cf-w" min="0" max="8" step="0.25" value="1"><span class="val" id="cf-wv">1.00</span></div>
      <div class="grp"><label><input type="checkbox" id="cf-interval"> interval CFG(只在 −4≤λ≤2 开引导)</label></div>
      <button class="primary" id="cf-run">采样 400 点</button>
    </div>
    <div class="grid2">
      <div><canvas id="cf-cv"></canvas></div>
      <div>
        <div class="readout" id="cf-read" style="min-height:170px">选 w,点「采样」。虚线圈 = 目标类的真实 1σ/2σ 等高线。</div>
        <div class="labnote">三个读数:<b>类中心偏距</b>(样本质心到真类均值的距离;注意方向是<em>径向向外</em>——朝远离其它 mode 的一侧);
        <b>类内离散度比</b>(vs 真条件分布;&gt;100% = 被推散/推出);<b>逃逸率</b>(跑到别的 mode 的比例——趋零是 CFG 买到的"服从性")。
        然后开「interval CFG」同 w 复测:两端关引导后形变减轻但不消失——production 里 interval guidance(SD3/Imagen 系)就是这笔买卖。</div>
      </div>
    </div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">完美模型,w=3,全程引导。类内 rms 离散度相对真条件分布会怎样?(常见教科书直觉:引导=锐化=收缩)</div>
    <div class="opts"><button data-opt="a">缩到 ~60%(教科书答案)</button><button data-opt="b">基本不变</button><button data-opt="c">反而涨过 120%,且整体外移</button></div>
    <div class="ans"><span class="verdict"></span> 我这台机器的实测(零模型误差):w=3 → 离散度比 <b>127%</b>,质心外偏 <b>0.37</b>;w=8 → <b>256%</b>,外偏 0.79。开 interval 后分别降到 111%/215%——缓解,不根治。
    机制如上:∇log p(c|x) 越过 mode 中心仍向外,引导把样本推到"比数据更像 c"的地方。所谓"多样性坍缩"发生在<b>选谁</b>维度(逃逸率→0,全部服从条件)与切向;"过曝"发生在<b>置信</b>方向(径向外推)。两件事同时发生,别混为一谈。
    坦白:我出题时预设的也是教科书答案 (a),被自己的实验打脸后才把机制想透——这道题连同这段话原样保留,作为"先预测再揭晓"的示范。生产对策:interval、rescale、autoguidance、把 w 蒸成输入(Flux)。</div>
  </div>

  <div class="lab">
    <div class="labhead"><span class="id">实验 6.2</span><span class="t">MNIST:w 扫描 · 像素统计</span><span class="pre">预渲染</span></div>
    <div class="imgrow" id="cf-mnist"></div>
    <div id="cf-sat-wrap"><canvas id="cf-sat"></canvas></div>
    <div class="labnote">图:同一批种子、digit 0/3/5/8,各 w 的 32 步采样。看 w 从 1→14:笔画变粗、变"标准体"、背景更黑、个体差异(斜的、歪的、断笔的)逐渐消失。
    曲线:|像素|&gt;0.95 的饱和占比随 w 单调上升——真实产品里的"CFG 过曝脸"在灰度 MNIST 上的对应物。生产参考值:SDXL 常用 w≈5–8,SD3/Flux(蒸馏后)有效 w 低得多。<span class="tag f">事实</span></div>
  </div>
  `;

  const plane = new Plane(document.getElementById('cf-cv'), { w: 500, h: 500, range: 2.6 });
  const el = id => document.getElementById(id);
  const K = R.gmm.K;

  function guided(modelName, cls, w, interval) {
    if (modelName === 'analytic') {
      return (x, lam, n) => {
        const useW = (!interval || (lam[0] >= -4 && lam[0] <= 2)) ? w : (w > 0 ? 1 : 0);
        if (useW === 0) return R.gmm.denoiseVP(x, lam, n);
        const dc = R.gmm.denoiseVP(x, lam, n, cls);
        if (useW === 1) return dc;
        const du = R.gmm.denoiseVP(x, lam, n);
        const out = new Float32Array(n * 2);
        for (let i = 0; i < n * 2; i++) out[i] = du[i] + useW * (dc[i] - du[i]);
        return out;
      };
    }
    const m = R.models.u_fm_cond;
    return (x, lam, n) => {
      const useW = (!interval || (lam[0] >= -4 && lam[0] <= 2)) ? w : (w > 0 ? 1 : 0);
      const clsArr = new Int32Array(n).fill(cls);
      if (useW === 0) return m.denoiseVP(x, lam, n, null);
      const dc = m.denoiseVP(x, lam, n, clsArr);
      if (useW === 1) return dc;
      const du = m.denoiseVP(x, lam, n, null);
      const out = new Float32Array(n * 2);
      for (let i = 0; i < n * 2; i++) out[i] = du[i] + useW * (dc[i] - du[i]);
      return out;
    };
  }

  function drawBase(cls) {
    plane.clear();
    plane.scatter(new Float32Array(R.meta.data_points.flat()), { color: '#39414E', r: 1.2, alpha: 0.3 });
    for (let m = 0; m < K; m++) plane.cross(R.gmm.mu[m][0], R.gmm.mu[m][1], { color: MODE_COLORS[m], size: 5 });
    const mu = R.gmm.mu[cls], sd = R.gmm.std[cls];
    plane.ring(mu[0], mu[1], sd, { color: MODE_COLORS[cls] });
    plane.ring(mu[0], mu[1], 2 * sd, { color: MODE_COLORS[cls] });
  }

  async function run() {
    const cls = +el('cf-cls').value, w = +el('cf-w').value;
    const modelName = el('cf-model').value, interval = el('cf-interval').checked;
    const N = 400, rng = mulberry32(97531);
    const x1 = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) x1[i] = gauss(rng);
    const s = fmEuler(guided(modelName, cls, w, interval), x1, N, 24);
    runToEnd(s);
    const xs = new Float32Array(N * 2);
    for (let i = 0; i < N * 2; i++) xs[i] = fmToVP(s.xs[i], 9);
    drawBase(cls);
    plane.scatter(xs, { color: C.fm, r: 2.4, alpha: 0.8 });
    // metrics
    const mu = R.gmm.mu[cls], sd = R.gmm.std[cls];
    let mx = 0, my = 0, esc = 0;
    for (let i = 0; i < N; i++) { mx += xs[i * 2] / N; my += xs[i * 2 + 1] / N; }
    let disp = 0, cnt = 0;
    for (let i = 0; i < N; i++) {
      const d = Math.hypot(xs[i * 2] - mu[0], xs[i * 2 + 1] - mu[1]);
      let dOther = 1e9;
      for (let k = 0; k < K; k++) if (k !== cls) dOther = Math.min(dOther, Math.hypot(xs[i * 2] - R.gmm.mu[k][0], xs[i * 2 + 1] - R.gmm.mu[k][1]));
      if (dOther < d) { esc++; continue; }
      disp += d * d; cnt++;
    }
    const rmsD = Math.sqrt(disp / Math.max(cnt, 1));
    const trueRms = sd * Math.SQRT2;  // isotropic 2D: E[r^2]=2sd^2
    el('cf-read').innerHTML =
      `模型=${modelName === 'analytic' ? '解析(零模型误差)' : '训练网络'}  w=<b>${w.toFixed(2)}</b>${interval ? '(区间)' : ''}\n` +
      `类中心偏距  = <b>${Math.hypot(mx - mu[0], my - mu[1]).toFixed(3)}</b>\n` +
      `类内 rms 半径 = <b>${rmsD.toFixed(3)}</b>  <span class="dim">真条件分布 = ${trueRms.toFixed(3)}</span>\n` +
      `离散度比    = <b>${(rmsD / trueRms * 100).toFixed(0)}%</b>  <span class="dim">(&gt;100% = 被推出真条件分布)</span>\n` +
      `逃逸率      = <b>${(esc / N * 100).toFixed(1)}%</b>\n` +
      `<span class="dim">w=0 无条件 / w=1 真条件 / w>1 越过数据向"更像 c"外推</span>`;
  }
  el('cf-run').addEventListener('click', run);
  el('cf-w').addEventListener('input', () => el('cf-wv').textContent = (+el('cf-w').value).toFixed(2));
  drawBase(0);

  // ---- MNIST assets ----
  try {
    const sat = await (await fetch('data/mnist/cfg.json')).json();
    const row = document.getElementById('cf-mnist');
    row.innerHTML = sat.map(s => `<div class="imgcell"><img src="data/mnist/cfg_${String(s.w).replace('.', 'p')}.png" width="200">
      <div class="cap">w=<b>${s.w}</b>${s.w === 0 ? '(无条件)' : s.w === 1 ? '(纯条件)' : ''}</div></div>`).join('');
    const ch = new Chart(document.getElementById('cf-sat'), { w: Math.min(760, mount.clientWidth - 20), h: 220, xlab: 'CFG 强度 w', ylab: '饱和像素占比', xlim: [0, 14.5], ylim: [0, Math.max(...sat.map(s => s.sat)) * 1.25 + 0.01] });
    ch.onRedraw(() => {
      ch.clear(); ch.axes();
      ch.line(sat.map(s => s.w), sat.map(s => s.sat), { color: C.warn, name: '|像素|>0.95 占比' });
      ch.dots(sat.map(s => s.w), sat.map(s => s.sat), { color: C.warn, r: 3.5 });
    });
    ch.redraw();
  } catch { document.getElementById('cf-mnist').innerHTML = '<div class="small">MNIST CFG 资产待生成。</div>'; }
}
