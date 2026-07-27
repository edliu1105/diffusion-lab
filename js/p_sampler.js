// p_sampler.js — §5: sampling = numerical integration of the learned field.
// Live particle arena, per-step readouts, NFE benchmarks, and two on-the-spot proofs.

import { R } from './models.js';
import { Plane, MODE_COLORS } from './viz2d.js';
import { C, Chart, legendHTML, lamColor } from './charts.js';
import { mulberry32, gauss } from './gmm.js';
import { fmEuler, ddim, edmHeun, runToEnd, endpointGap, fmToVP, veToVP } from './diff.js';

const SAMPLERS = {
  ddpm: { label: 'DDPM ancestral(η=1)', color: C.ddpm, mk: (d, n, steps, rng) => ddim(d, initVP(n, rng), n, steps, { eta: 1, rng }) },
  ddim: { label: 'DDIM(η=0,确定性)', color: C.ddim, mk: (d, n, steps, rng) => ddim(d, initVP(n, rng), n, steps, { eta: 0 }) },
  heun: { label: 'EDM-Heun(2 阶)', color: C.edm, mk: (d, n, steps, rng) => edmHeun(d, initVE(n, rng), n, Math.max(1, Math.round(steps / 2)), {}) },
  churn: { label: 'EDM-Heun + churn', color: C.edm, dash: [5, 4], mk: (d, n, steps, rng) => edmHeun(d, initVE(n, rng), n, Math.max(1, Math.round(steps / 2)), { churn: 8, rng }) },
  fme: { label: 'FM-Euler(1 阶)', color: C.fm, mk: (d, n, steps, rng) => fmEuler(d, initN(n, rng), n, steps) },
};
function initN(n, rng) { const x = new Float32Array(n * 2); for (let i = 0; i < n * 2; i++) x[i] = gauss(rng); return x; }
function initVP(n, rng) { return initN(n, rng); }                       // lam_min: x ~ N(0,I)
function initVE(n, rng) { const x = initN(n, rng); for (let i = 0; i < n * 2; i++) x[i] *= 8.0; return x; }
const toVP = { fm: fmToVP, ve: veToVP, vp: (x) => x };

export async function init(mount) {
  mount.innerHTML = `
  <p>训练完,你手里只有一个场 D(x,λ)。生成 = 从纯噪声出发,沿这个场往 λ 增大的方向<em>积分</em>。所有采样器只在三个旋钮上不同:</p>
  <ul>
    <li><b>网格</b>:在 λ 轴上落哪些步点(均匀 λ?Karras 的 σ^{1/ρ} 网格?FM 的均匀 t?——都只是同一根轴的不同刻度);</li>
    <li><b>阶数</b>:每步用几次前向去估计局部斜率(Euler=1,Heun=2,DPM-Solver++ 等=多点外推);</li>
    <li><b>随机性</b>:走的时候要不要重新掺一点噪声再去噪(η、churn——SDE 和 ODE 的差别仅此而已)。</li>
  </ul>
  <p class="small">术语注:<b>ancestral sampling</b> 是从概率图模型继承的老词——有向图上按"祖先先于后代"的顺序逐节点抽样。放到 DDPM 就是:先采链根 x_T~N(0,I),再沿反向链每步从 p(x_{t−1}|x_t) <em>真抽一个样本</em>(每步注入新噪声),而不是取均值。名字说的是采样顺序,与"祖先"再无其他关系;口语里就叫 DDPM 采样。</p>
  <p>DDPM 的 ancestral sampling 不是"另一种算法",它就是 η=1 的 DDIM,也是 reverse SDE 的一种离散化;DDIM 是 η=0 的确定性极限,也是 probability-flow ODE 的 exponential integrator。下面亲手跑。</p>

  <div class="lab">
    <div class="labhead"><span class="id">实验 5.1</span><span class="t">粒子竞技场 · 逐步读数</span><span class="live">LIVE</span></div>
    <div class="ctl">
      <div class="grp"><label>场</label><select id="sp-model">
        <option value="u_fm">u_fm(训练网络)</option>
        <option value="analytic">解析真值(完美模型)</option>
        <option value="eps_ddpm">eps_ddpm</option>
        <option value="x0_edm">x0_edm</option>
      </select></div>
      <div class="grp"><label>采样器</label><select id="sp-sampler">
        ${Object.entries(SAMPLERS).map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('')}
      </select></div>
      <div class="grp"><label>步数(NFE 预算)</label><input type="range" id="sp-steps" min="1" max="6" step="1" value="4"><span class="val" id="sp-stepsv">16</span></div>
    </div>
    <div class="ctl">
      <button class="primary" id="sp-play">▶ 播放</button>
      <button id="sp-step">单步 →</button>
      <button id="sp-reset">重置(新种子)</button>
      <div class="grp"><label><input type="checkbox" id="sp-trails" checked> 轨迹尾迹</label></div>
    </div>
    <div class="grid2">
      <div><canvas id="sp-cv"></canvas></div>
      <div>
        <div class="readout" id="sp-read" style="min-height:150px">按「播放」。粒子颜色 = 当前 λ 的光谱色。</div>
        <div class="readout" id="sp-metrics" style="margin-top:10px;min-height:64px"></div>
        <div class="labnote">读数里的 <b>|Δx|</b> 是这一步的平均移动距离——注意它随 λ 的分布:开头几步移动巨大(定构图),后面步子越来越小(修细节)。<b>把步数拉到 2,再看每步读数</b>:同样的场,预算怎么花完全变了。</div>
      </div>
    </div>
  </div>

  <div class="challenge" data-correct="b">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">用 u_fm + FM-Euler,把步数从 16 降到 2。端点分布会先失去什么?</div>
    <div class="opts">
      <button data-opt="a">六个 mode 的位置(整体构图)</button>
      <button data-opt="b">mode 内的紧致度(局部细节),点会系统性偏向 mode 之间/中心</button>
      <button data-opt="c">随机崩坏,无规律</button>
    </div>
    <div class="ans"><span class="verdict"></span> 少步 Euler 的误差来自<b>轨迹曲率</b>:场在高噪段指向"数据均值",到低噪段才分岔指向具体 mode。步子太大,粒子带着"均值方向"的惯性冲过了分岔口,落点被系统性拉向内侧/mode 间——这不是随机误差,是有方向的偏置。构图级的信息(往哪个 mode 去)由最早几步决定,反而最抗少步。这就是 §7 蒸馏要解决的问题:把弯路学直。</div>
  </div>

  <div class="lab">
    <div class="labhead"><span class="id">实验 5.2</span><span class="t">NFE 基准:步数 ↓ 质量曲线</span><span class="live">LIVE·点击运行</span></div>
    <div class="ctl">
      <div class="grp"><label>场</label><select id="nb-model"><option value="u_fm">u_fm</option><option value="analytic">解析真值</option></select></div>
      <button class="primary" id="nb-run">跑基准(≈5s)</button><span class="val" id="nb-prog"></span>
    </div>
    <div id="nb-legend"></div>
    <canvas id="nb-conv"></canvas>
    <canvas id="nb-pure" style="margin-top:12px"></canvas>
    <div class="labnote" id="nb-note">上图:到各自 256 步参考解的端点距离(离散化误差,log-log)。看斜率:Euler 族 ~N⁻¹,Heun ~N⁻²——这就是"高阶求解器"四个字的全部含义。
    下图:落点纯度(到最近 mode 中心的平均距离)。注意两图分歧处:<b>离散化误差小 ≠ 样本好</b>,ODE 解得再准,极限也只是"模型的分布",不是数据的分布。模型误差和积分误差是两笔账。</div>
  </div>

  <h3>[证明] 两个当场完成的数值等价</h3>
  <div class="grid2">
    <div class="lab" style="margin-top:0">
      <div class="labhead"><span class="id">证明 5.3</span><span class="t">DDIM ≡ FM-Euler(同一条 ODE)</span><span class="live">LIVE</span></div>
      <button class="primary" id="pf-run1">运行</button>
      <div class="readout" id="pf-out1" style="margin-top:10px;min-height:130px">同 64 份噪声种子,DDIM(VP 坐标)与 FM-Euler(直线坐标)分别积分,端点换算到同一坐标系后测中位距离。步数加倍,距离应按一阶收敛速率缩小,趋于同一条曲线的同一端点。</div>
    </div>
    <div class="lab" style="margin-top:0">
      <div class="labhead"><span class="id">证明 5.4</span><span class="t">DDPM = DDIM + 噪声(η 连续插值)</span><span class="live">LIVE</span></div>
      <button class="primary" id="pf-run2">运行</button>
      <div class="readout" id="pf-out2" style="margin-top:10px;min-height:130px">固定种子与步数,η 从 0 → 1 扫描:每步注入 τ(η) 的新噪声、确定性部分相应缩减。η=1 恰好还原教科书 DDPM ancestral sampling 的方差。分布统计(均值/方差/纯度)应全程稳定——<b>随机性改变路径,不改变终点分布</b>(模型完美时)。</div>
    </div>
  </div>
  <div id="sp-mnist"></div>
  `;

  // ---------------- arena ----------------
  const plane = new Plane(document.getElementById('sp-cv'), { w: 520, h: 520, range: 2.6 });
  const el = id => document.getElementById(id);
  const stepsOf = v => [1, 2, 4, 8, 16, 32, 64][v - 1] ?? 16;
  let arena = null, playing = false;

  function denoiserFor(name) {
    if (name === 'analytic') return (x, lam, n) => R.gmm.denoiseVP(x, lam, n);
    const m = R.models[name];
    return (x, lam, n) => m.denoiseVP(x, lam, n);
  }
  function makeArena() {
    const seed = (Math.random() * 1e9) | 0;
    const rng = mulberry32(seed);
    const sKey = el('sp-sampler').value;
    const steps = stepsOf(+el('sp-steps').value);
    const N = 320;
    const s = SAMPLERS[sKey].mk(denoiserFor(el('sp-model').value), N, steps, rng);
    return { s, N, sKey, trails: [], lastLam: null, seed };
  }
  function drawArena(readout = null) {
    plane.clear();
    plane.scatter(new Float32Array(R.meta.data_points.flat()), { color: '#3A4250', r: 1.2, alpha: 0.35 });
    for (let m = 0; m < R.gmm.K; m++) plane.cross(R.gmm.mu[m][0], R.gmm.mu[m][1], { color: MODE_COLORS[m], size: 5 });
    if (!arena) return;
    const { s, N } = arena;
    const lam = arena.lastLam ?? -8.5;
    if (el('sp-trails').checked) {
      for (const t of arena.trails) plane.path(t.pts, { color: t.color, width: 0.7, alpha: 0.35 });
    }
    const conv = toVP[s.frame];
    const xy = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) { xy[i * 2] = conv(s.xs[i * 2], lam); xy[i * 2 + 1] = conv(s.xs[i * 2 + 1], lam); }
    plane.scatter(xy, { color: lamColor(Math.max(-9, Math.min(10, lam))), r: 2.2, alpha: 0.85 });
    if (readout) el('sp-read').innerHTML = readout;
  }
  function stepArena() {
    if (!arena || arena.s.done) return false;
    const { s } = arena;
    const prev = Float32Array.from(s.xs);
    const prevLam = arena.lastLam ?? -8.5;
    const r = s.step();
    if (!r) return false;
    arena.lastLam = r.lam;
    const conv = toVP[s.frame];
    // record short trails for 40 tracked particles
    const track = 40;
    for (let i = 0; i < track; i++) {
      if (!arena.trails[i]) arena.trails[i] = { pts: [], color: 'rgba(154,163,178,1)' };
      const t = arena.trails[i];
      if (t.pts.length === 0) t.pts.push(conv(prev[i * 2], prevLam), conv(prev[i * 2 + 1], prevLam));
      t.pts.push(conv(s.xs[i * 2], r.lam), conv(s.xs[i * 2 + 1], r.lam));
    }
    const nfe = s.nfePerStep ? s.k * s.nfePerStep : s.k;
    drawArena(
      `<b>step ${s.k}/${s.steps}</b>   <span class="dim">NFE=${nfe}</span>\n` +
      `${r.label}\n` +
      `λ = <b>${r.lam.toFixed(2)}</b>  <span class="dim">(光谱位置见粒子颜色)</span>\n` +
      `平均 |Δx| = <b>${r.dxRms.toFixed(4)}</b>\n` +
      `<span class="dim">x̂₀ 的 rms = ${rms(r.x0h).toFixed(3)}(网络此刻对终点的猜测)</span>`);
    if (s.done) finishArena();
    return true;
  }
  function rms(a) { let s2 = 0; for (const v of a) s2 += v * v; return Math.sqrt(s2 / a.length); }
  function purity(xs, n) {
    let acc = 0;
    for (let i = 0; i < n; i++) {
      let best = 1e9;
      for (let k = 0; k < R.gmm.K; k++) {
        const d = Math.hypot(xs[i * 2] - R.gmm.mu[k][0], xs[i * 2 + 1] - R.gmm.mu[k][1]);
        if (d < best) best = d;
      }
      acc += best;
    }
    return acc / n;
  }
  function finishArena() {
    const { s, N } = arena;
    const conv = toVP[s.frame];
    const xy = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) { xy[i * 2] = conv(s.xs[i * 2], 9); xy[i * 2 + 1] = conv(s.xs[i * 2 + 1], 9); }
    el('sp-metrics').innerHTML = `<b>终点指标</b>  落点纯度(→0 好)= <b>${purity(xy, N).toFixed(4)}</b>` +
      `   <span class="dim">数据自身 = ${purity(new Float32Array(R.meta.data_points.flat()), R.meta.data_points.length).toFixed(4)}(理想参照:mode 内自然散布)</span>`;
  }
  async function play() {
    if (playing) { playing = false; el('sp-play').textContent = '▶ 播放'; return; }
    if (!arena || arena.s.done) arena = makeArena();
    playing = true; el('sp-play').textContent = '⏸ 暂停';
    while (playing && stepArena()) await new Promise(r => setTimeout(r, 90));
    playing = false; el('sp-play').textContent = '▶ 播放';
  }
  el('sp-play').addEventListener('click', play);
  el('sp-step').addEventListener('click', () => { if (!arena || arena.s.done) arena = makeArena(); stepArena(); });
  el('sp-reset').addEventListener('click', () => { playing = false; arena = makeArena(); el('sp-metrics').innerHTML = ''; drawArena('已重置。'); });
  el('sp-steps').addEventListener('input', () => { el('sp-stepsv').textContent = stepsOf(+el('sp-steps').value); });
  ['sp-model', 'sp-sampler'].forEach(id => el(id).addEventListener('change', () => { playing = false; arena = null; drawArena(); }));
  drawArena();

  // ---------------- NFE benchmark ----------------
  const wCh = Math.min(860, mount.clientWidth - 20);
  el('nb-legend').innerHTML = legendHTML(Object.values(SAMPLERS).map(v => [v.label, v.color, !!v.dash]));
  el('nb-run').addEventListener('click', async () => {
    const model = el('nb-model').value;
    const d = denoiserFor(model);
    const NFES = [2, 4, 8, 16, 32];
    const N = 220, seed = 777;
    const res = {};
    let done = 0, total = Object.keys(SAMPLERS).length * (NFES.length + 1);
    for (const [key, S] of Object.entries(SAMPLERS)) {
      // reference: same family, 256 NFE, same seeds
      const ref = S.mk(d, N, 256, mulberry32(seed));
      runToEnd(ref);
      const refVP = frameToVP(ref);
      el('nb-prog').textContent = `${++done}/${total}`;
      await tick();
      res[key] = [];
      for (const nfe of NFES) {
        const s = S.mk(d, N, nfe, mulberry32(seed));
        runToEnd(s);
        const xs = frameToVP(s);
        res[key].push({ nfe: s.nfePerStep ? s.steps * s.nfePerStep : s.steps, gap: endpointGap(xs, refVP, N), pure: purity(xs, N) });
        el('nb-prog').textContent = `${++done}/${total}`;
        await tick();
      }
    }
    const conv = new Chart(el('nb-conv'), { w: wCh, h: 270, xlab: 'NFE(前向次数,log2)', ylab: '到参考解距离(log)', xlim: [0.9, 5.3], ylim: [1e-4, 2], ylog: true });
    conv.onRedraw(() => {
      conv.clear(); conv.axes({ xticks: [1, 2, 3, 4, 5], xfmt: v => String(2 ** v) });
      for (const [key, S] of Object.entries(SAMPLERS)) {
        if (key === 'ddpm' || key === 'churn') continue;  // stochastic: no fixed reference point
        conv.line(res[key].map(r => Math.log2(r.nfe)), res[key].map(r => r.gap), { color: S.color, name: S.label });
        conv.dots(res[key].map(r => Math.log2(r.nfe)), res[key].map(r => r.gap), { color: S.color, r: 3 });
      }
    });
    conv.redraw();
    const pure = new Chart(el('nb-pure'), { w: wCh, h: 270, xlab: 'NFE(log2)', ylab: '落点纯度(→数据参照虚线)', xlim: [0.9, 5.3], ylim: [0, 0.9] });
    const dataP = purity(new Float32Array(R.meta.data_points.flat()), R.meta.data_points.length);
    pure.onRedraw(() => {
      pure.clear(); pure.axes({ xticks: [1, 2, 3, 4, 5], xfmt: v => String(2 ** v) });
      pure.line([0.9, 5.3], [dataP, dataP], { color: C.ink3, dash: [5, 4], width: 1, name: '数据自身' });
      for (const [key, S] of Object.entries(SAMPLERS)) {
        pure.line(res[key].map(r => Math.log2(r.nfe)), res[key].map(r => r.pure), { color: S.color, name: S.label, dash: S.dash || [] });
        pure.dots(res[key].map(r => Math.log2(r.nfe)), res[key].map(r => r.pure), { color: S.color, r: 3 });
      }
    });
    pure.redraw();
    el('nb-prog').textContent = '完成';
  });
  function frameToVP(s) {
    const conv = toVP[s.frame];
    const out = new Float32Array(s.xs.length);
    for (let i = 0; i < s.xs.length; i++) out[i] = conv(s.xs[i], 9);
    return out;
  }
  const tick = () => new Promise(r => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });

  // ---------------- proofs ----------------
  el('pf-run1').addEventListener('click', async () => {
    const d = denoiserFor('analytic');
    const N = 64, seed = 4242;
    const lines = [];
    for (const steps of [8, 32, 128, 400]) {
      const a = ddim(d, initVP(N, mulberry32(seed)), N, steps, { eta: 0, lamMin: -8.5, lamMax: 9 });
      runToEnd(a);
      const b = fmEuler(d, initN(N, mulberry32(seed)), N, steps, { tStart: 0.9858350, tEnd: 0.0110090 }); // t=sigmoid(±λ/2):与 DDIM 的 λ∈[−8.5,9] 端点精确对齐
      runToEnd(b);
      const bVP = new Float32Array(N * 2);
      for (let i = 0; i < N * 2; i++) bVP[i] = fmToVP(b.xs[i], 9);
      const g = endpointGap(a.xs, bVP, N);
      lines.push(`${String(steps).padStart(4)} 步:中位端点距离 = <b>${g.toFixed(5)}</b>`);
      el('pf-out1').innerHTML = lines.join('\n') + '\n<span class="dim">计算中…</span>';
      await tick();
    }
    lines.push(`<span class="dim">──────────</span>`,
      `两种"算法"在无限步极限走的是<b>同一条曲线</b>。`,
      `差异随步数按 ~1/N 消失 = 纯离散化差异。`,
      `DDIM(2020)与 FM-Euler(2023)相差的只是坐标与网格。`);
    el('pf-out1').innerHTML = lines.join('\n');
  });

  // ---------------- MNIST grids ----------------
  try {
    await (await fetch('data/mnist/grids.json')).json();
    const NFES = [1, 2, 4, 8, 16, 32];
    const SLIST = [['fme', 'FM-Euler'], ['ddim', 'DDIM'], ['ddpm', 'DDPM(η=1)'], ['heun', 'EDM-Heun']];
    el('sp-mnist').innerHTML = `
    <div class="lab">
      <div class="labhead"><span class="id">实验 5.5</span><span class="t">MNIST:同种子 · 采样器 × NFE</span><span class="pre">预渲染</span></div>
      <div class="ctl">
        <div class="grp"><label>左</label><select id="mg-a">${SLIST.map(([k, l]) => `<option value="${k}">${l}</option>`).join('')}</select></div>
        <div class="grp"><label>右</label><select id="mg-b">${SLIST.map(([k, l], i) => `<option value="${k}" ${i === 2 ? 'selected' : ''}>${l}</option>`).join('')}</select></div>
        <div class="grp"><label>NFE</label><input type="range" id="mg-nfe" min="0" max="5" step="1" value="3"><span class="val" id="mg-nfev">8</span></div>
      </div>
      <div class="imgrow">
        <div class="imgcell"><img id="mg-img-a" width="300"><div class="cap" id="mg-cap-a"></div></div>
        <div class="imgcell"><img id="mg-img-b" width="300"><div class="cap" id="mg-cap-b"></div></div>
      </div>
      <div class="labnote">64 份固定种子(行=数字 0–7)。值得做的三个对照:
      ① FM-Euler 的 NFE=1→4:一步是"平均数字"的糊影(§4 挑战 B 的那团均值),4 步已经"选好了字",之后只是变利索——<b>构图信息花的是最早几步</b>;
      ② 同 NFE=8 比 DDIM vs DDPM:注噪版在低预算下明显更毛糙(每步的新噪声来不及被后续步收拾);
      ③ Heun 在 NFE=4(即 2 大步)时的断笔——二阶法步子太少时插值假设崩坏,<b>高阶≠少步之王,蒸馏才是</b>。</div>
    </div>`;
    const upd = () => {
      const nfe = NFES[+el('mg-nfe').value];
      el('mg-nfev').textContent = nfe;
      for (const side of ['a', 'b']) {
        const sel2 = el('mg-' + side);
        el(`mg-img-${side}`).src = `data/mnist/grid_${sel2.value}_${nfe}.png`;
        el(`mg-cap-${side}`).innerHTML = `<b>${sel2.selectedOptions[0].text}</b> · NFE=${nfe}`;
      }
    };
    ['mg-a', 'mg-b', 'mg-nfe'].forEach(id => el(id).addEventListener(id === 'mg-nfe' ? 'input' : 'change', upd));
    upd();
    // per-step tensor strip (inference microscope, image version)
    try {
      const tr = await (await fetch('data/mnist/infer_trace.json')).json();
      el('sp-mnist').insertAdjacentHTML('beforeend', `
      <div class="lab">
        <div class="labhead"><span class="id">实验 5.6</span><span class="t">MNIST 推理显微镜:8 步 × 3 张量</span><span class="pre">预渲染·真实张量</span></div>
        <div class="imgcell"><img src="data/mnist/infer_trace.png" style="width:100%;max-width:820px"><div class="cap">行 1:x_t(当前状态)| 行 2:x̂₀(此刻对终点的最优猜测)| 行 3:û(此刻的速度场输出)。列=步。</div></div>
        <div style="overflow-x:auto"><table class="data"><tr><th>步</th>${tr.map((s, i) => `<th>${i + 1}</th>`).join('')}</tr>
          <tr><td class="m">t</td>${tr.map(s => `<td class="num">${s.t.toFixed(2)}</td>`).join('')}</tr>
          <tr><td class="m">λ</td>${tr.map(s => `<td class="num">${s.lam.toFixed(1)}</td>`).join('')}</tr>
          <tr><td class="m">std(x_t)</td>${tr.map(s => `<td class="num">${s.x_std.toFixed(2)}</td>`).join('')}</tr>
          <tr><td class="m">std(x̂₀)</td>${tr.map(s => `<td class="num">${s.x0h_std.toFixed(2)}</td>`).join('')}</tr>
          <tr><td class="m">rms(û)</td>${tr.map(s => `<td class="num">${s.u_rms.toFixed(2)}</td>`).join('')}</tr></table></div>
        <div class="labnote">看第二行:x̂₀ 从"类均值 blob"起步,第 2–3 步就定下这个数字的骨架,之后八成的步数都花在锐化上——<b>构图早、细节晚</b>,和 2D 粒子的 |Δx| 分布、和 §3 的 λ-频率对应,是同一件事的三个视角。std(x̂₀) 随步数单调上升(从均值的低方差爬向真实数据的方差)也是值得背下来的形状。</div>
      </div>`);
    } catch { }
  } catch { el('sp-mnist').innerHTML = '<div class="small">MNIST 网格资产待生成(训练完成后自动出现)。</div>'; }

  el('pf-run2').addEventListener('click', async () => {
    const d = denoiserFor('analytic');
    const N = 400, steps = 64, seed = 999;
    const lines = [`η 扫描,${steps} 步,${N} 粒子,同种子:`];
    for (const eta of [0, 0.25, 0.5, 1.0]) {
      const s = ddim(d, initVP(N, mulberry32(seed)), N, steps, { eta, rng: mulberry32(seed + 1) });
      runToEnd(s);
      const p = purity(s.xs, N);
      let m0 = 0, v0 = 0;
      for (let i = 0; i < N; i++) { m0 += s.xs[i * 2] / N; }
      for (let i = 0; i < N; i++) { v0 += (s.xs[i * 2] - m0) ** 2 / N; }
      lines.push(`η=${eta.toFixed(2)}  纯度=<b>${p.toFixed(4)}</b>  E[x]=${m0.toFixed(3)}  Var[x]=${v0.toFixed(3)}`);
      el('pf-out2').innerHTML = lines.join('\n');
      await tick();
    }
    lines.push(`<span class="dim">──────────</span>`,
      `每条路径完全不同(η>0 时逐步注噪),`,
      `但端点的分布统计一致 —— <b>SDE 与 ODE 共享全部边缘分布</b>,`,
      `这就是"ancestral sampling、DDIM、probability-flow ODE 是一家"的操作性含义。`);
    el('pf-out2').innerHTML = lines.join('\n');
  });
}
