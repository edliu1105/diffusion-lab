// p_anatomy.js — §4: what each prediction target LOOKS like on a real UNet, per lambda.
// Assets prerendered by exp/mnist_export.py (real tensors, no illustrations).

import { C, drawLamStrip } from './charts.js';
import { vpFromLam, fmTFromLam } from './nn.js';

const ROWS = [
  ['x_t', 'x_t = (1−t)·x₀ + t·ε', '网络的输入。往左拖:图像淹没进噪声的全过程。'],
  ['x0h', 'x̂₀ = x_t − t·û', '网络对原图的最优猜测。高噪端它不是垃圾——是「类条件均值」:一团平均数字。'],
  ['epsh', 'ε̂ = (x_t − (1−t)·x̂₀)/t', '对掺入噪声的猜测。低噪端它趋向纯白噪(没信息可提取时,猜测≈输入的残差)。'],
  ['vh', 'v̂ = α·ε̂ − σ·x̂₀', '两端都有界。看它连续地从「像 ε」旋转成「像 −x₀」。'],
  ['uh', 'û = ε̂ − x̂₀', 'FM 速度:恒为「噪声图减数字图」,全程 O(1)——这就是采样器直接吃它的原因。'],
  ['score', 'score = −ε̂/σ', '注意每格下方的显示尺度:右端尺度爆炸(×10²)。任何直接回归它的尝试都会死在这里。'],
];

export async function init(mount) {
  let A;
  try { A = await (await fetch('data/mnist/anatomy.json')).json(); }
  catch { mount.innerHTML = `<div class="callout">MNIST 资产尚未生成(训练进行中)。稍后刷新。</div>`; return; }

  const CELL = 32, GAP = 2, SC = 3;   // sprite geometry & upscale
  mount.innerHTML = `
  <p>读论文时那些符号——ε̂、x̂₀、v̂、û、score——在一个真实 UNet 里各自是一张什么样的图?
  下面全部来自本机刚训好的 4.5M 参数 flow-matching UNet 对同一张数字、同一份噪声、不同 λ 的前向输出(±仿射换算)。
  <strong>先自己预测,再拖滑杆。</strong>目标手感:给定 λ,你能画出六张图的草图、报出它们的 std。</p>

  <div class="lab">
    <div class="labhead"><span class="id">实验 4.1</span><span class="t">六张量 × 噪声光谱</span><span class="pre">预渲染·真实张量</span></div>
    <div class="ctl">
      <div class="grp"><label>λ</label><input type="range" id="an-slider" min="0" max="${A.lams.length - 1}" step="1" value="5">
      <canvas class="lamstrip" id="an-strip"></canvas><span class="val" id="an-lamv"></span></div>
      <div class="readout" id="an-coef" style="padding:4px 10px"></div>
    </div>
    <div class="flex">
      <div class="imgcell"><img id="an-x0ref" src="data/mnist/anatomy_x0.png" width="${CELL * SC}"><div class="cap">真 x₀(digit ${A.digit})</div></div>
      <div class="imgcell"><img id="an-epsref" src="data/mnist/anatomy_eps.png" width="${CELL * SC}"><div class="cap">真 ε(固定)</div></div>
      <div style="flex:1"></div>
    </div>
    <div class="flex" id="an-cells"></div>
    <div class="labnote">每格下方:该张量的实测 std 与显示映射范围(超出被裁剪)。score 行的范围是自适应的——注意它涨得多快。</div>
  </div>

  <div class="challenge" data-correct="c">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">λ = 0(信噪各半)。v̂ = α·ε̂ − σ·x̂₀ 这张图长什么样?</div>
    <div class="opts">
      <button data-opt="a">基本就是 ε̂(噪声占主导)</button>
      <button data-opt="b">基本就是 −x̂₀(图像负片)</button>
      <button data-opt="c">噪声底子上叠着一张负片数字,五五开</button>
    </div>
    <div class="ans"><span class="verdict"></span> λ=0 时 α=σ=0.707,两项等权:v̂ = 0.707(ε̂ − x̂₀)。拖到 λ=0 亲自看——噪声纹理里嵌着一个暗色的数字轮廓。
    理论 std 也能心算:Var(v)=α²·Var(ε)+σ²·Var(x₀)=0.5·1+0.5·${(A.data_pixel_std ** 2).toFixed(2)}≈${(0.5 + 0.5 * A.data_pixel_std ** 2).toFixed(2)},std≈${Math.sqrt(0.5 + 0.5 * A.data_pixel_std ** 2).toFixed(2)}(实测见格子下方)。<b>能这样心算张量的 std,是"手感"的最小单元</b>:任何时候你打印一个中间张量,都应该先知道它该多大。</div>
  </div>

  <div class="challenge" data-correct="b">
    <div class="chead">⚡ 预测挑战</div>
    <div class="q">把 λ 拖到 −8(几乎纯噪声)。x̂₀ 那格会是什么?</div>
    <div class="opts">
      <button data-opt="a">纯灰(网络放弃)</button>
      <button data-opt="b">一团模糊的"平均数字"</button>
      <button data-opt="c">随机的另一个数字</button>
    </div>
    <div class="ans"><span class="verdict"></span> 条件期望的本能:没有信息时,最优 L2 猜测=先验均值。这个网络是类条件的,所以是"这个数字类的平均像"。
    <b>整条生成轨迹就是这团均值 blob 被逐步锐化成一个具体样本的过程</b>——§5 的推理轨迹条能看到 x̂₀ 从 blob 变清晰的全程。顺带:这也是为什么少步生成必然牺牲多样性锐度,以及为什么"回归教师端点"的朴素蒸馏会糊(§7)。</div>
  </div>

  <h4>极限行为对照表(退化方向背下来,调试时直接用)</h4>
  <table class="data">
    <tr><th></th><th>λ → −∞(纯噪)</th><th>λ → +∞(干净)</th><th>std 的解析式(每像素)</th></tr>
    <tr><td class="m">x̂₀</td><td>E[x₀|c](均值 blob)</td><td>x_t 本身</td><td class="m">≤ σ_data</td></tr>
    <tr><td class="m">ε̂</td><td>≈ x_t(整图就是噪声)</td><td>不可辨识,趋于噪声/伪影</td><td class="m">≈ 1</td></tr>
    <tr><td class="m">v̂</td><td>→ ε 方向</td><td>→ −x₀ 方向</td><td class="m">√(α²+σ²·σ²_data)</td></tr>
    <tr><td class="m">û</td><td>→ ε − E[x₀]</td><td>→ ε̂ − x₀</td><td class="m">√(1+σ²_data) 级</td></tr>
    <tr><td class="m">score</td><td>≈ −x_t(高斯先验的 score)</td><td>发散 ~e^{λ/2}</td><td class="m">1/σ</td></tr>
  </table>
  <p class="small">「ε̂ 在低噪端不可辨识」值得咀嚼:λ=+9 时 x_t 里噪声份额 σ≈0.011,任何 ε̂ 误差对重建都无所谓(×σ 衰减),loss 给不出压强,于是网络输出什么都行——这就是 ε-模型在最后几步经常输出结构化伪影、而且没人在乎的原因;但把它换算成别的头用时(比如接 DDIM 的 ε̂ 项)就会咬人。</p>
  `;

  const slider = document.getElementById('an-slider');
  const strip = document.getElementById('an-strip');
  const cells = document.getElementById('an-cells');
  const sprite = new Image(); sprite.src = 'data/mnist/anatomy.png';
  // NOT decode(): hidden tabs may defer decode() indefinitely; onload always fires.
  await new Promise((res, rej) => {
    if (sprite.complete && sprite.naturalWidth) return res();
    sprite.onload = res; sprite.onerror = () => rej(new Error('anatomy sprite failed to load'));
  });

  function render() {
    const i = +slider.value;
    const lam = A.lams[i], st = A.stats[i];
    document.getElementById('an-lamv').textContent = 'λ=' + lam.toFixed(0);
    drawLamStrip(strip, -9, 10, lam);
    const { a, s } = vpFromLam(lam);
    document.getElementById('an-coef').innerHTML =
      `t=<b>${fmTFromLam(lam).toFixed(3)}</b>  α=<b>${a.toFixed(3)}</b>  σ=<b>${s.toFixed(3)}</b>  ` +
      `<span class="dim">x̂₀ 的重建误差 mse=${st.x0h_mse.toExponential(1)}</span>`;
    cells.innerHTML = ROWS.map(([key, formula, note], r) => {
      const cv = `<canvas width="${CELL}" height="${CELL}" data-r="${r}" data-c="${i}" style="width:${CELL * SC}px;height:${CELL * SC}px;image-rendering:pixelated;border:1px solid var(--line2);border-radius:4px;background:#000"></canvas>`;
      const info = st[key];
      return `<div class="imgcell" style="max-width:${CELL * SC + 16}px">${cv}
        <div class="cap"><b>${formula}</b><br>std=<b>${info.std.toFixed(2)}</b> · 显示±${info.disp_range.toFixed(1)}<br><span style="color:var(--ink3)">${note}</span></div></div>`;
    }).join('');
    for (const cv of cells.querySelectorAll('canvas')) {
      const r = +cv.dataset.r, c = +cv.dataset.c;
      const ctx = cv.getContext('2d');
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(sprite, c * (CELL + 2), r * (CELL + 2), CELL, CELL, 0, 0, CELL, CELL);
    }
  }
  slider.addEventListener('input', render);
  render();
}
