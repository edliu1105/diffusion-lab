// charts.js — hand-rolled canvas chart kit. Dark surface, recessive grid,
// mono numerals, hover crosshair. One categorical system for the whole lab:
// lineage colors follow the ENTITY (DDPM/DDIM/EDM/FM/student/analytic), never the slot.

// Series palette validated (dataviz six checks, dark surface #161B23):
// order [ddpm, edm, ddim, fm, student]; student↔fm sits in the CVD warn band,
// so student series always carry a dash as secondary encoding.
export const C = {
  bg: '#101318', panel: '#161B23', panel2: '#1B212B',
  ink: '#E8E4DA', ink2: '#9AA3B2', ink3: '#5C6675',
  grid: 'rgba(154,163,178,0.10)', axis: 'rgba(154,163,178,0.35)',
  analytic: '#E8E4DA',
  ddpm: '#5083DC',   // eps / ancestral lineage
  ddim: '#A26BE4',   // deterministic VP sampler (VP lineage, distinct hue)
  edm: '#C28324',    // Karras lineage
  fm: '#D65077',     // flow matching lineage
  student: '#1FA9A3',// distilled (dash when co-plotted with fm)
  ok: '#79C99E', warn: '#E2B33C', bad: '#E06C5F',
};

// logSNR spectrum: noise end (violet) -> data end (amber). The lab's spine.
const LAM_STOPS = [
  [-9, [110, 116, 238]], [-4, [122, 130, 210]], [0, [150, 148, 160]],
  [4, [210, 165, 92]], [10, [242, 176, 61]],
];
export function lamColor(lam, alpha = 1) {
  let i = 0;
  while (i < LAM_STOPS.length - 2 && lam > LAM_STOPS[i + 1][0]) i++;
  const [l0, c0] = LAM_STOPS[i], [l1, c1] = LAM_STOPS[i + 1];
  const f = Math.min(1, Math.max(0, (lam - l0) / (l1 - l0)));
  const c = c0.map((v, k) => Math.round(v + f * (c1[k] - v)));
  return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

export function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const FONT_M = '11px Consolas, "Cascadia Mono", monospace';
const FONT_S = '10px Consolas, "Cascadia Mono", monospace';

export class Chart {
  constructor(canvas, { w = 560, h = 300, ml = 52, mr = 14, mt = 14, mb = 34,
                        xlab = '', ylab = '', xlim = [0, 1], ylim = [0, 1], ylog = false } = {}) {
    this.cv = canvas; this.w = w; this.h = h;
    this.ml = ml; this.mr = mr; this.mt = mt; this.mb = mb;
    this.xlab = xlab; this.ylab = ylab; this.xlim = xlim; this.ylim = ylim; this.ylog = ylog;
    this.ctx = setupCanvas(canvas, w, h);
    this.series = [];       // for hover: {name, color, xs, ys}
    this._hover = null;
    canvas.addEventListener('mousemove', e => this._onMove(e));
    canvas.addEventListener('mouseleave', () => { this._hover = null; this.redraw(); });
  }
  X(v) { return this.ml + (v - this.xlim[0]) / (this.xlim[1] - this.xlim[0]) * (this.w - this.ml - this.mr); }
  Y(v) {
    if (this.ylog) v = Math.log10(Math.max(v, 1e-12));
    const [a, b] = this.ylog ? this.ylim.map(x => Math.log10(Math.max(x, 1e-12))) : this.ylim;
    return this.h - this.mb - (v - a) / (b - a) * (this.h - this.mt - this.mb);
  }
  clear() {
    const { ctx } = this;
    ctx.clearRect(0, 0, this.w, this.h);
    this.series = [];
  }
  axes({ xticks = null, yticks = null, xfmt = v => fmt(v), yfmt = v => fmt(v) } = {}) {
    const { ctx } = this;
    ctx.font = FONT_S; ctx.fillStyle = C.ink3;
    ctx.strokeStyle = C.grid; ctx.lineWidth = 1;
    const xs = xticks || autoTicks(this.xlim[0], this.xlim[1], 7);
    const ys = yticks || (this.ylog ? logTicks(this.ylim[0], this.ylim[1]) : autoTicks(this.ylim[0], this.ylim[1], 5));
    for (const t of xs) {
      const x = this.X(t);
      ctx.beginPath(); ctx.moveTo(x, this.mt); ctx.lineTo(x, this.h - this.mb); ctx.stroke();
      ctx.textAlign = 'center'; ctx.fillText(xfmt(t), x, this.h - this.mb + 14);
    }
    for (const t of ys) {
      const y = this.Y(t);
      ctx.beginPath(); ctx.moveTo(this.ml, y); ctx.lineTo(this.w - this.mr, y); ctx.stroke();
      ctx.textAlign = 'right'; ctx.fillText(yfmt(t), this.ml - 6, y + 3.5);
    }
    ctx.strokeStyle = C.axis;
    ctx.strokeRect(this.ml, this.mt, this.w - this.ml - this.mr, this.h - this.mt - this.mb);
    ctx.fillStyle = C.ink2; ctx.font = FONT_M;
    if (this.xlab) { ctx.textAlign = 'center'; ctx.fillText(this.xlab, (this.ml + this.w - this.mr) / 2, this.h - 4); }
    if (this.ylab) {
      ctx.save(); ctx.translate(12, (this.mt + this.h - this.mb) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = 'center'; ctx.fillText(this.ylab, 0, 0); ctx.restore();
    }
  }
  line(xs, ys, { color = C.ink, width = 2, dash = [], name = null, alpha = 1 } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = alpha;
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    let started = false;
    for (let i = 0; i < xs.length; i++) {
      if (ys[i] == null || !isFinite(ys[i])) { started = false; continue; }
      const x = this.X(xs[i]), y = this.Y(ys[i]);
      if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.restore();
    if (name) this.series.push({ name, color, xs, ys });
  }
  dots(xs, ys, { color = C.ink, r = 2.5, name = null, alpha = 1 } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = alpha; ctx.fillStyle = color;
    for (let i = 0; i < xs.length; i++) {
      if (!isFinite(ys[i])) continue;
      ctx.beginPath(); ctx.arc(this.X(xs[i]), this.Y(ys[i]), r, 0, 7); ctx.fill();
    }
    ctx.restore();
    if (name) this.series.push({ name, color, xs, ys, dots: true });
  }
  vline(x, { color = C.ink3, dash = [4, 4], width = 1 } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(this.X(x), this.mt); ctx.lineTo(this.X(x), this.h - this.mb); ctx.stroke();
    ctx.restore();
  }
  label(x, y, text, { color = C.ink2, align = 'left', font = FONT_S } = {}) {
    const { ctx } = this;
    ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align;
    ctx.fillText(text, this.X(x), this.Y(y));
  }
  // hover: crosshair + tooltip listing every named series at nearest x
  onRedraw(fn) { this._redraw = fn; }
  redraw() { if (this._redraw) { this._redraw(); this._drawHover(); } }
  _onMove(e) {
    const rect = this.cv.getBoundingClientRect();
    const px = e.clientX - rect.left;
    if (px < this.ml || px > this.w - this.mr) { this._hover = null; this.redraw(); return; }
    this._hover = this.xlim[0] + (px - this.ml) / (this.w - this.ml - this.mr) * (this.xlim[1] - this.xlim[0]);
    this.redraw();
  }
  _drawHover() {
    if (this._hover == null || !this.series.length) return;
    const { ctx } = this;
    const hx = this._hover;
    ctx.save();
    ctx.strokeStyle = 'rgba(232,228,218,0.35)'; ctx.setLineDash([3, 3]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(this.X(hx), this.mt); ctx.lineTo(this.X(hx), this.h - this.mb); ctx.stroke();
    const rows = [];
    for (const s of this.series) {
      let bi = -1, bd = Infinity;
      for (let i = 0; i < s.xs.length; i++) {
        const d = Math.abs(s.xs[i] - hx);
        if (d < bd && isFinite(s.ys[i]) && s.ys[i] != null) { bd = d; bi = i; }
      }
      if (bi >= 0) rows.push({ name: s.name, color: s.color, x: s.xs[bi], y: s.ys[bi] });
    }
    if (rows.length) {
      ctx.font = FONT_S;
      const wBox = 10 + Math.max(...rows.map(r => ctx.measureText(`${r.name} ${fmt(r.y)}`).width)) + 16;
      const hBox = rows.length * 15 + 22;
      let bx = this.X(hx) + 10;
      if (bx + wBox > this.w - this.mr) bx = this.X(hx) - wBox - 10;
      const by = this.mt + 6;
      ctx.fillStyle = 'rgba(16,19,24,0.92)'; ctx.strokeStyle = C.axis;
      ctx.beginPath(); ctx.roundRect(bx, by, wBox, hBox, 4); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.ink2; ctx.textAlign = 'left';
      ctx.fillText(`${this.xlab || 'x'} = ${fmt(rows[0].x)}`, bx + 8, by + 14);
      rows.forEach((r, i) => {
        ctx.fillStyle = r.color;
        ctx.fillRect(bx + 8, by + 22 + i * 15 - 6, 7, 7);
        ctx.fillStyle = C.ink;
        ctx.fillText(`${r.name}  ${fmt(r.y)}`, bx + 19, by + 22 + i * 15);
      });
    }
    ctx.restore();
  }
}

export function fmt(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  return v.toExponential(1);
}
export function autoTicks(a, b, n) {
  const span = b - a, step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n + 1) || mag * 10;
  const t0 = Math.ceil(a / step) * step, out = [];
  for (let t = t0; t <= b + 1e-9; t += step) out.push(Math.abs(t) < 1e-10 ? 0 : t);
  return out;
}
export function logTicks(a, b) {
  const out = [];
  for (let e = Math.ceil(Math.log10(Math.max(a, 1e-12))); 10 ** e <= b * 1.0001; e++) out.push(10 ** e);
  return out;
}

// HTML legend with fixed entity colors (color follows entity, never slot)
export function legendHTML(items) {
  return `<div class="legend">` + items.map(([name, color, dash]) =>
    `<span class="lg"><i style="background:${dash ? 'transparent' : color};border-top:${dash ? `2px dashed ${color}` : 'none'}"></i>${name}</span>`).join('') + `</div>`;
}

// the lam spectrum strip used by every noise-level slider
export function drawLamStrip(canvas, lamMin, lamMax, cur = null) {
  const w = canvas.clientWidth || 300, h = 10;
  const ctx = setupCanvas(canvas, w, h);
  for (let px = 0; px < w; px++) {
    const lam = lamMin + (lamMax - lamMin) * px / w;
    ctx.fillStyle = lamColor(lam);
    ctx.fillRect(px, 2, 1, 6);
  }
  if (cur != null) {
    const x = (cur - lamMin) / (lamMax - lamMin) * w;
    ctx.fillStyle = C.ink; ctx.fillRect(x - 1, 0, 2, 10);
  }
}
