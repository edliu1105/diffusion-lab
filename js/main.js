// main.js — boot: TOC + scrollspy, challenge component, lam rail, lazy panel init.

import { drawLamStrip } from './charts.js';
import { load } from './models.js';

// ---- TOC ----
const toc = document.getElementById('toc');
const secs = [...document.querySelectorAll('section.ch')];
for (const s of secs) {
  const a = document.createElement('a');
  a.href = '#' + s.id;
  const n = s.querySelector('.eyebrow')?.textContent.split('·')[0].trim() || '';
  a.innerHTML = `<span class="n">${n}</span>${s.querySelector('h2').textContent}`;
  toc.appendChild(a);
}
const spy = new IntersectionObserver(es => {
  for (const e of es) if (e.isIntersecting) {
    toc.querySelectorAll('a').forEach(a => a.classList.toggle('on', a.hash === '#' + e.target.id));
  }
}, { rootMargin: '-20% 0px -70% 0px' });
secs.forEach(s => spy.observe(s));

drawLamStrip(document.getElementById('lamrail'), -9, 10);

// ---- challenge component ----
// <div class="challenge" data-q="..."><button data-opt>..</button>... <div class="ans">..</div></div>
export function wireChallenges(root = document) {
  for (const ch of root.querySelectorAll('.challenge:not([data-wired])')) {
    ch.dataset.wired = '1';
    const ans = ch.querySelector('.ans');
    const opts = ch.querySelectorAll('[data-opt]');
    opts.forEach(b => b.addEventListener('click', () => {
      opts.forEach(o => o.classList.remove('picked'));
      b.classList.add('picked');
      if (ans) {
        ans.classList.add('show');
        const v = ans.querySelector('.verdict');
        if (v && b.dataset.opt) {
          const right = b.dataset.opt === (ch.dataset.correct || '');
          v.innerHTML = (ch.dataset.correct ? (right ? '✓ 你答对了。' : `✗ 你选了「${b.textContent.trim()}」。`) : '');
        }
      }
    }));
    const rv = ch.querySelector('[data-reveal]');
    if (rv && ans) rv.addEventListener('click', () => ans.classList.add('show'));
  }
}

// ---- panels ----
const PANELS = [
  ['./p_hero.js', null],
  ['./p_s0.js', 'c-s0'],
  ['./p_s1.js', 'c-s1'],
  ['./p_field.js', 'c-s2'],
  ['./p_train.js', 'c-s3'],
  ['./p_anatomy.js', 'c-s4'],
  ['./p_sampler.js', 'c-s5'],
  ['./p_cfg.js', 'c-s6'],
  ['./p_distill.js', 'c-s7'],
  ['./p_s8.js', 'c-s8'],
  ['./p_s9.js', 'c-s9'],
  ['./p_museum.js', 'c-s10'],
  ['./p_s11.js', 'c-s11'],
];

(async () => {
  try { await load(); } catch (e) {
    console.error(e);
    document.getElementById('c-s0').innerHTML =
      `<div class="callout">数据加载失败(${e.message})。请通过本地 HTTP 服务打开(python -m http.server),file:// 无法 fetch。</div>`;
    return;
  }
  for (const [mod, mount] of PANELS) {
    try {
      const m = await import(mod);
      await m.init(mount ? document.getElementById(mount) : null);
    } catch (e) {
      console.error(`panel ${mod} failed:`, e);
      if (mount) {
        const el = document.getElementById(mount);
        if (el && !el.children.length) el.innerHTML = `<div class="small">此面板加载失败:<code>${mod}</code> — ${e.message}</div>`;
      }
    }
  }
  wireChallenges();
})();
