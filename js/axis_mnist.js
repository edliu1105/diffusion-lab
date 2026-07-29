// axis_mnist.js — probe B: the same lambda cursor, read out as real UNet tensors.
// All pixels come from app/data/mnist/* (rendered by the real 4.5M-param model earlier).

import { S } from './axis_main.js';
import { vpFromLam, fmTFromLam } from './nn.js';
import { AMP } from './axis_main.js';
import { lamColor } from './charts.js';

const CELL = 32, GAP = 2;
const ROWKEY = ['x_t', 'x0h', 'epsh', 'vh', 'uh', 'score'];
const ROWLAB = {
  x_t: ['x_t', '网络输入'], x0h: ['x̂₀', 'x_t − t·û'], epsh: ['ε̂', '(x_t−(1−t)x̂₀)/t'],
  vh: ['v̂', 'α·ε̂ − σ·x̂₀'], uh: ['û', 'ε̂ − x̂₀'], score: ['score', '−ε̂/σ'],
};
const HEAD2ROW = { x0: 'x0h', D: 'x0h', eps: 'epsh', v: 'vh', u: 'uh', score: 'score' };

let A = null, sprite = null, trace = null, traceImg = null;
let host, footEl, tiles = {}, stripCv, stripMark, pairDiv, traceDiv, traceCv;
let sfDiv, sfX0 = null, sfEps = null, sfMask = null, sfRho = 0.4, sfSeed = 7;

export async function initMnist(hostEl, foot) {
  host = hostEl; footEl = foot;
  A = await (await fetch('data/mnist/anatomy.json')).json();
  sprite = new Image(); sprite.src = 'data/mnist/anatomy.png';
  await new Promise(res => { sprite.onload = res; });
  try {
    trace = await (await fetch('data/mnist/infer_trace.json')).json();
    traceImg = new Image(); traceImg.src = 'data/mnist/infer_trace.png';
    await new Promise(res => { traceImg.onload = res; traceImg.onerror = res; });
  } catch { }

  const SC = 3;
  const tilesRow = document.createElement('div'); tilesRow.className = 'mn-row'; tilesRow.id = 'mn-tiles';
  for (const key of ROWKEY) {
    const cell = document.createElement('div'); cell.className = 'mn-cell'; cell.dataset.k = key;
    const cvs = document.createElement('canvas');
    cvs.width = CELL; cvs.height = CELL;
    cvs.style.width = cvs.style.height = CELL * SC + 'px';
    const cap = document.createElement('div'); cap.className = 'mn-cap';
    cell.append(cvs, cap); tilesRow.append(cell);
    tiles[key] = { cvs, cap, cell };
  }
  // noise->digit strip (row 0 across all lambda columns)
  const stripWrap = document.createElement('div'); stripWrap.id = 'mn-strip';
  stripCv = document.createElement('canvas');
  const SW = 42;
  stripCv.width = A.lams.length * (SW + 3) - 3; stripCv.height = SW + 14;
  stripCv.style.width = stripCv.width + 'px';
  stripWrap.append(stripCv);
  stripCv.addEventListener('click', e => {
    const r = stripCv.getBoundingClientRect();
    const j = Math.max(0, Math.min(A.lams.length - 1, Math.floor((e.clientX - r.left) / (SW + 3))));
    import('./axis_main.js').then(m => m.setS({ lam: A.lams[j] }));
  });
  // Self-Flow: per-token dual-lambda input construction (computed live from the real x0 & eps pixels)
  sfDiv = document.createElement('div');
  sfDiv.innerHTML = `
    <div class="mn-cap" style="text-align:left;margin:2px 0 6px">
      <b>Self-Flow 视角(卡②)</b>:整图一个 λ → 每个 token 自己的 λ。上下文 token 停在数据端(λ=+9),其余跟随主游标。</div>
    <div class="mn-row">
      <div class="mn-cell"><canvas id="sf-map" width="32" height="32" style="width:96px;height:96px"></canvas><div class="mn-cap">token-λ 地图<br>(颜色 = 主轴光谱)</div></div>
      <div class="mn-cell"><canvas id="sf-mix" width="32" height="32" style="width:96px;height:96px"></canvas><div class="mn-cap">双 λ 混合输入<br><b id="sf-stat"></b></div></div>
      <div style="flex:1;min-width:170px">
        <div class="mn-cap" style="text-align:left">上下文比例 ρ <input type="range" id="sf-rho" min="0" max="0.9" step="0.05" value="0.4" style="width:100px;vertical-align:middle"> <span id="sf-rhov">0.40</span>
        <button id="sf-roll" style="font-size:11px;padding:2px 8px;margin-left:6px">重掷 mask</button></div>
        <div class="mn-cap" style="text-align:left;color:var(--ink3)">要推断噪 token,读干净 token 是唯一出路——信息不对称即表征监督。此处展示的是<b>输入构造本身</b>(真像素真公式);表征收益见论文训练对照。</div>
      </div>
    </div>`;

  // distill pair
  pairDiv = document.createElement('div'); pairDiv.id = 'mn-pair'; pairDiv.style.display = 'none';
  pairDiv.innerHTML = `
    <div class="mn-cell"><img src="data/mnist/student_teacher.png" width="272" style="image-rendering:pixelated;border:1px solid var(--line2);border-radius:5px"><div class="mn-cap"><b>教师</b> · 32 NFE</div></div>
    <div class="mn-cell"><img src="data/mnist/student_1step.png" width="272" style="image-rendering:pixelated;border:1px solid var(--line2);border-radius:5px"><div class="mn-cap"><b>一步学生</b> · 1 NFE(同种子;高频能量 88%)</div></div>`;
  // inference trace (8-step real run)
  traceDiv = document.createElement('div'); traceDiv.style.display = 'none';
  traceCv = document.createElement('canvas');
  traceDiv.append(traceCv);
  const capT = document.createElement('div'); capT.className = 'mn-cap';
  capT.textContent = '一次真实 8 步 FM-Euler:上 x_t,下 x̂₀(与轴上行走器对齐)';
  traceDiv.append(capT);

  host.append(tilesRow, stripWrap, sfDiv, traceDiv, pairDiv);

  // decode the real x0 / eps pixels once (inverting the export display mappings)
  const grab = (src, lo, hi) => {
    const img = new Image(); img.src = src;
    return new Promise(res => {
      img.onload = () => {
        const c = document.createElement('canvas'); c.width = 32; c.height = 32;
        const g = c.getContext('2d');
        g.drawImage(img, 0, 0);
        const d = g.getImageData(0, 0, 32, 32).data;
        const out = new Float32Array(32 * 32);
        for (let i = 0; i < 1024; i++) out[i] = lo + d[i * 4] / 255 * (hi - lo);
        res(out);
      };
      img.onerror = () => res(null);
    });
  };
  [sfX0, sfEps] = await Promise.all([
    grab('data/mnist/anatomy_x0.png', -1, 1),
    grab('data/mnist/anatomy_eps.png', -3, 3),
  ]);
  rollMask();
  document.getElementById('sf-rho').addEventListener('input', e => {
    sfRho = +e.target.value;
    document.getElementById('sf-rhov').textContent = sfRho.toFixed(2);
    rollMask(); drawSelfFlow();
  });
  document.getElementById('sf-roll').addEventListener('click', () => { sfSeed++; rollMask(); drawSelfFlow(); });
}

function rollMask() {
  // deterministic per-token context mask, 8x8 tokens of 4x4 px
  let s = sfSeed >>> 0;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  sfMask = new Uint8Array(64);
  for (let k = 0; k < 64; k++) sfMask[k] = rnd() < sfRho ? 1 : 0;
}

function drawSelfFlow() {
  if (!sfX0 || !sfEps || !sfMask) return;
  const LAM_CTX = 9;
  const tOf = l => 1 / (1 + Math.exp(l / 2));
  const tCtx = tOf(LAM_CTX), tNoisy = tOf(S.lam);
  const map = document.getElementById('sf-map').getContext('2d');
  const mix = document.getElementById('sf-mix').getContext('2d');
  const im = mix.createImageData(32, 32);
  for (let p = 0; p < 1024; p++) {
    const px = p % 32, py = (p / 32) | 0;
    const tok = ((py >> 2) << 3) + (px >> 2);
    const t = sfMask[tok] ? tCtx : tNoisy;
    const v = (1 - t) * sfX0[p] + t * sfEps[p];              // the actual dual-lambda input
    const u8 = Math.max(0, Math.min(255, (v + 1.6) / 3.2 * 255));
    im.data[p * 4] = im.data[p * 4 + 1] = im.data[p * 4 + 2] = u8;
    im.data[p * 4 + 3] = 255;
  }
  mix.putImageData(im, 0, 0);
  for (let tok = 0; tok < 64; tok++) {
    map.fillStyle = lamColor(sfMask[tok] ? LAM_CTX : S.lam);
    map.fillRect((tok % 8) * 4, (tok >> 3) * 4, 4, 4);
  }
  const nCtx = sfMask.reduce((a, b) => a + b, 0);
  document.getElementById('sf-stat').textContent = `${64 - nCtx} 个 token @ λ=${S.lam.toFixed(1)} · ${nCtx} 个上下文 @ λ=9`;
}

function colBlend(lam) {
  const L = A.lams;
  let j = 0;
  while (j < L.length - 2 && lam > L[j + 1]) j++;
  const f = Math.max(0, Math.min(1, (lam - L[j]) / (L[j + 1] - L[j])));
  return [j, f];
}
const lerp = (a, b, f) => a + (b - a) * f;

export function drawMnist() {
  if (!A || !sprite) return;
  const distill = S.toyView === 'distill';
  document.getElementById('mn-tiles').style.display = distill ? 'none' : '';
  stripCv.parentElement.style.display = distill ? 'none' : '';
  if (sfDiv) sfDiv.style.display = distill ? 'none' : '';
  if (!distill) drawSelfFlow();
  pairDiv.style.display = distill ? '' : 'none';
  traceDiv.style.display = (!distill && S.walkK > 0 && traceImg && traceImg.width) ? '' : 'none';
  if (distill) {
    footEl.innerHTML = `与探头 A 同一件事的图像版:教师沿轴走 32 步;一步学生(MeanFlow 意义下的平均速度)把整根轴压成 1 次前向。同种子逐列对比:结构一致,损失的是笔画锐度(高频能量 88%)。`;
    return;
  }

  const [j, f] = colBlend(S.lam);
  const st0 = A.stats[j], st1 = A.stats[Math.min(j + 1, A.stats.length - 1)];
  const selRow = HEAD2ROW[S.head];
  ROWKEY.forEach((key, r) => {
    const { cvs, cap, cell } = tiles[key];
    const g = cvs.getContext('2d');
    g.imageSmoothingEnabled = false;
    g.clearRect(0, 0, CELL, CELL);
    g.globalAlpha = 1;
    g.drawImage(sprite, j * (CELL + GAP), r * (CELL + GAP), CELL, CELL, 0, 0, CELL, CELL);
    if (f > 0.02 && j + 1 < A.lams.length) {
      g.globalAlpha = f;
      g.drawImage(sprite, (j + 1) * (CELL + GAP), r * (CELL + GAP), CELL, CELL, 0, 0, CELL, CELL);
      g.globalAlpha = 1;
    }
    const std = lerp(st0[key].std, st1[key].std, f);
    const rng = lerp(st0[key].disp_range, st1[key].disp_range, f);
    const [nm, formula] = ROWLAB[key];
    cap.innerHTML = `<b>${nm}</b> ${key === 'x_t' ? '' : '<span style="opacity:.7">' + formula + '</span>'}<br>std <b>${std.toFixed(2)}</b> · 显示±${rng.toFixed(1)}`;
    cell.classList.toggle('hl', key === selRow);
  });

  // strip with cursor
  const SW = 42, g = stripCv.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.clearRect(0, 0, stripCv.width, stripCv.height);
  A.lams.forEach((l, k) => {
    g.drawImage(sprite, k * (CELL + GAP), 0, CELL, CELL, k * (SW + 3), 12, SW, SW);
    g.fillStyle = '#5C6675'; g.font = '9px Consolas'; g.textAlign = 'center';
    g.fillText(String(l), k * (SW + 3) + SW / 2, 9);
  });
  const mx = (j + f) * (SW + 3) + SW / 2;
  g.fillStyle = '#F2B03D';
  g.beginPath(); g.moveTo(mx, 12); g.lineTo(mx - 5, 2); g.lineTo(mx + 5, 2); g.closePath(); g.fill();

  // inference trace strip
  if (S.mode === 'infer' && traceImg && traceImg.width && trace) {
    const n = trace.length, TW = 40;
    traceCv.width = n * (TW + 3) - 3; traceCv.height = TW * 2 + 4;
    traceCv.style.width = traceCv.width + 'px';
    const tg = traceCv.getContext('2d');
    tg.imageSmoothingEnabled = false;
    // rows in infer_trace.png: 0=x_t, 1=x0h
    for (let k = 0; k < n; k++) {
      tg.drawImage(traceImg, k * (CELL + GAP), 0, CELL, CELL, k * (TW + 3), 0, TW, TW);
      tg.drawImage(traceImg, k * (CELL + GAP), (CELL + GAP), CELL, CELL, k * (TW + 3), TW + 4, TW, TW);
    }
    // walker marker: nearest trace step by lambda
    let best = 0, bd = 1e9;
    trace.forEach((tr, k) => { const d = Math.abs(tr.lam - S.lam); if (d < bd) { bd = d; best = k; } });
    tg.strokeStyle = '#F2B03D'; tg.lineWidth = 2;
    tg.strokeRect(best * (TW + 3) + 1, 1, TW - 2, TW * 2 + 2);
  }

  // foot
  const { a, s } = vpFromLam(S.lam), t = fmTFromLam(S.lam);
  const amp = AMP.find(x => x.key === S.head).f(S.lam);
  const selStd = lerp(st0[selRow].std, st1[selRow].std, f);
  footEl.innerHTML =
    `λ=<b>${S.lam.toFixed(2)}</b> → t=<b>${t.toFixed(3)}</b> α=${a.toFixed(3)} σ=${s.toFixed(3)} · ` +
    `当前衣服 <b>${S.head}</b>:std=<b>${selStd.toFixed(2)}</b>,1 份误差换 x̂₀ 误差 ×<b>${amp.toPrecision(3)}</b>(见轴底条件数泳道)` +
    `${S.head === 'score' ? ' · score 的显示范围在右端爆炸——没人裸奔回归它' : ''}`;
}
