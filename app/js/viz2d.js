// viz2d.js — the 2D world<->pixel plane every toy panel draws on. VP frame.

import { setupCanvas, C } from './charts.js';

export class Plane {
  constructor(canvas, { w = 520, h = 520, range = 2.6 } = {}) {
    this.cv = canvas; this.w = w; this.h = h; this.range = range;
    this.ctx = setupCanvas(canvas, w, h);
    canvas.classList.add('fieldcv');
  }
  px(x) { return (x / this.range * 0.5 + 0.5) * this.w; }
  py(y) { return (-y / this.range * 0.5 + 0.5) * this.h; }
  wx(px) { return (px / this.w - 0.5) * 2 * this.range; }
  wy(py) { return -(py / this.h - 0.5) * 2 * this.range; }
  clear() {
    const { ctx } = this;
    ctx.fillStyle = '#0B0E12'; ctx.fillRect(0, 0, this.w, this.h);
    ctx.strokeStyle = 'rgba(154,163,178,0.08)'; ctx.lineWidth = 1;
    for (let g = -2; g <= 2; g++) {
      ctx.beginPath(); ctx.moveTo(this.px(g), 0); ctx.lineTo(this.px(g), this.h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, this.py(g)); ctx.lineTo(this.w, this.py(g)); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(154,163,178,0.16)';
    ctx.beginPath(); ctx.moveTo(this.px(0), 0); ctx.lineTo(this.px(0), this.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, this.py(0)); ctx.lineTo(this.w, this.py(0)); ctx.stroke();
  }
  scatter(xy, { color = C.ink2, r = 1.6, alpha = 0.5, colors = null } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = alpha;
    const n = xy.length / 2;
    for (let i = 0; i < n; i++) {
      ctx.fillStyle = colors ? colors[i] : color;
      ctx.beginPath(); ctx.arc(this.px(xy[i * 2]), this.py(xy[i * 2 + 1]), r, 0, 7); ctx.fill();
    }
    ctx.restore();
  }
  arrow(x0, y0, dx, dy, { color = C.ink, width = 1.5, alpha = 0.9, head = 4 } = {}) {
    const { ctx } = this;
    const ax = this.px(x0), ay = this.py(y0);
    const bx = this.px(x0 + dx), by = this.py(y0 + dy);
    const L = Math.hypot(bx - ax, by - ay);
    if (L < 0.75) return;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    const ang = Math.atan2(by - ay, bx - ax), hs = Math.min(head, L * 0.5);
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(bx - hs * Math.cos(ang - 0.45), by - hs * Math.sin(ang - 0.45));
    ctx.lineTo(bx - hs * Math.cos(ang + 0.45), by - hs * Math.sin(ang + 0.45));
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }
  cross(x, y, { color = C.ink, size = 5, width = 1.5 } = {}) {
    const { ctx } = this;
    const px = this.px(x), py = this.py(y);
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width;
    ctx.beginPath(); ctx.moveTo(px - size, py); ctx.lineTo(px + size, py);
    ctx.moveTo(px, py - size); ctx.lineTo(px, py + size); ctx.stroke();
    ctx.restore();
  }
  ring(x, y, r, { color = C.ink2, width = 1, dash = [3, 3] } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath(); ctx.arc(this.px(x), this.py(y), r / this.range * this.w / 2, 0, 7); ctx.stroke();
    ctx.restore();
  }
  path(pts, { color = C.ink, width = 1.2, alpha = 0.8, dash = [] } = {}) {
    const { ctx } = this;
    ctx.save(); ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash);
    ctx.beginPath();
    for (let i = 0; i < pts.length / 2; i++) {
      const x = this.px(pts[i * 2]), y = this.py(pts[i * 2 + 1]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke(); ctx.restore();
  }
  text(x, y, s, { color = C.ink2, font = '11px Consolas, monospace', align = 'left' } = {}) {
    const { ctx } = this;
    ctx.font = font; ctx.fillStyle = color; ctx.textAlign = align;
    ctx.fillText(s, this.px(x), this.py(y));
  }
}

// mode colors for the 6-GMM (categorical, muted; identity of modes, fixed)
export const MODE_COLORS = ['#8FA8D8', '#C39ED0', '#D89AA6', '#CBB183', '#9DBE93', '#84BBB8'];
