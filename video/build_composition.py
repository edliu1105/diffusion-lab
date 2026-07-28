"""Emit index.html + assets/js/data_timing.js from SCRIPT.md + vo/manifest.json.
All data-* timing baked statically (contract: root duration read pre-script)."""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = json.load(open(os.path.join(HERE, 'assets', 'vo', 'manifest.json')))

SCENES = [
    ('s0', '', '', ['s0a', 's0b'], 0.5, 1.2),
    ('s1', '§1 · 一根轴', 'λ = logSNR:全课唯一的横轴', ['s1a', 's1b', 's1c', 's1d', 's1e'], 1.0, 1.4),
    ('s2', '§2 · 五种读数', '一个条件期望,五件衣服', ['s2a', 's2b', 's2c', 's2d', 's2e'], 1.0, 1.4),
    ('s3', '§3 · 训练显微镜', '六行循环 · λ 预算 · 贝叶斯地板', ['s3a', 's3b', 's3c', 's3d'], 1.0, 1.4),
    ('s4', '§4 · 无网络生成', '闭式贝叶斯 · 复读机 · 背叛', ['s4a', 's4b', 's4c', 's4d', 's4e'], 1.0, 1.4),
    ('s5', '§5 · 采样 = 数值积分', '一条 ODE,几把尺子,一笔随机预算', ['s5a', 's5b', 's5c', 's5d'], 1.0, 1.4),
    ('s6', '§6 · CFG', '分布的算术,与它的账单', ['s6a', 's6b', 's6c', 's6d'], 1.0, 1.4),
    ('s7', '§7 · 蒸馏', '换学习对象:从方向场到终点映射', ['s7a', 's7b', 's7c', 's7d'], 1.0, 1.4),
    ('s8', '§8 · 把时间接上', '因果化 · 疫苗 · 世界模型', ['s8a', 's8b', 's8c', 's8d', 's8e'], 1.0, 1.4),
    ('s9', '§9 · 定稿', '五个心智模型', ['s9a', 's9b', 's9c'], 1.0, 2.0),
]
GAP = 0.9

def seg_text():
    txt = {}
    for line in open(os.path.join(HERE, 'SCRIPT.md'), encoding='utf-8'):
        m = re.match(r'^- (s\w+) \| (.+)$', line.strip())
        if m: txt[m.group(1)] = m.group(2).strip()
    return txt

TXT = seg_text()
timing, t = {}, 0.0
for sid, eyebrow, title, segs, lead, tail in SCENES:
    start = t
    t += lead
    ss = []
    for g in segs:
        d = MAN[g]['dur']
        ss.append({'id': g, 'start': round(t, 3), 'dur': round(d, 3)})
        t += d + GAP
    t += tail - GAP
    timing[sid] = {'start': round(start, 3), 'dur': round(t - start, 3), 'segs': ss}
TOTAL = round(t + 0.6, 3)
print(f'total video: {TOTAL/60:.2f} min')

open(os.path.join(HERE, 'assets', 'js', 'data_timing.js'), 'w').write(
    'window.HF_TIMING = ' + json.dumps({'TOTAL': TOTAL, **timing}) + ';\n')

# ------------------------------------------------- html pieces
def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')

audio_els, caption_els, scene_els = [], [], []
for sid, eyebrow, title, segs, lead, tail in SCENES:
    T = timing[sid]
    chrome = ''
    if title:
        chrome = (f'<div class="eyebrow" id="{sid}-eyebrow">{esc(eyebrow)}</div>'
                  f'<h1 class="title" id="{sid}-title">{esc(title)}</h1>')
    extra = ''
    if sid == 's0':
        extra = ('<div id="s0-big"><div id="s0-line1">整个领域只学一个对象</div>'
                 '<div id="s0-line2">E[ x<sub>0</sub> | x<sub>t</sub> ]</div></div>')
    if sid == 's3':
        lines = ['for step in range(N):',
                 '    x0   = sample_data(B)',
                 '    λ    = sample_noise_level()      # ① 预算',
                 '    ε    = randn_like(x0)',
                 '    x_t  = α(λ)·x0 + σ(λ)·ε',
                 '    loss = w(λ)·‖net(x_t,λ) − target‖²   # ② 汇率']
        spans = ''.join(f'<div class="codeline" id="s3-code-{i}">{esc(l)}</div>' for i, l in enumerate(lines))
        extra = f'<div id="s3-code">{spans}</div>'
    scene_els.append(
        f'<section class="clip scene" id="sc-{sid}" data-start="{T["start"]}" '
        f'data-duration="{T["dur"]}" data-track-index="2">{chrome}{extra}</section>')
    for s in T['segs']:
        audio_els.append(
            f'<audio id="vo-{s["id"]}" src="assets/vo/{s["id"]}.mp3" data-start="{s["start"]}" '
            f'data-duration="{s["dur"]}" data-track-index="10" data-volume="1"></audio>')
        caption_els.append(
            f'<div class="clip cap" id="cap-{s["id"]}" data-start="{s["start"]}" data-duration="{round(s["dur"]+0.55,3)}" '
            f'data-track-index="3"><span>{esc(TXT[s["id"]])}</span></div>')

HTML = '''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=1920, height=1080" />
<title>扩散模型:一个对象,一根轴</title>
<script src="assets/js/gsap.min.js"></script>
<script src="assets/js/mathlib.js"></script>
<script src="assets/js/data_models.js"></script>
<script src="assets/js/data_mnist.js"></script>
<script src="assets/js/data_drift.js"></script>
<script src="assets/js/data_timing.js"></script>
<script src="assets/js/scenes_core.js"></script>
<script src="assets/js/scenes_late.js"></script>
<style>
  @font-face { font-family: 'Noto Sans SC'; src: url('assets/fonts/NotoSansSC-Regular.otf'); font-weight: 400; }
  @font-face { font-family: 'Noto Sans SC'; src: url('assets/fonts/NotoSansSC-Bold.otf'); font-weight: 700; }
  @font-face { font-family: 'JetBrains Mono'; src: url('assets/fonts/JetBrainsMono-Regular.ttf'); font-weight: 400; }
  @font-face { font-family: 'JetBrains Mono'; src: url('assets/fonts/JetBrainsMono-Bold.ttf'); font-weight: 700; }
  body { margin: 0; background: #0C0F14; font-family: 'Noto Sans SC', sans-serif; }
  #root { position: relative; width: 1920px; height: 1080px; overflow: hidden; background: #0C0F14; }
  #bgfill { position: absolute; inset: 0; background:
      radial-gradient(1100px 700px at 30% 20%, rgba(110,116,238,0.07), transparent 60%),
      radial-gradient(1000px 700px at 78% 82%, rgba(242,176,61,0.05), transparent 60%), #0C0F14; }
  #maincv { position: absolute; inset: 0; }
  .scene { position: absolute; inset: 0; pointer-events: none; }
  .eyebrow { position: absolute; top: 74px; left: 120px; font: 24px 'JetBrains Mono', monospace;
             letter-spacing: 0.3em; color: #D89A3A; }
  .title { position: absolute; top: 104px; left: 118px; margin: 0; font-size: 54px; font-weight: 700;
           color: #E8E4DA; letter-spacing: 0.04em; }
  .cap { position: absolute; left: 0; right: 0; bottom: 0; height: 150px; display: flex;
         align-items: center; justify-content: center; }
  .cap span { display: inline-block; max-width: 1560px; padding: 14px 34px; font-size: 33px; line-height: 1.5;
         color: #E8E4DA; background: rgba(12,15,20,0.72); border: 1px solid rgba(154,163,178,0.18);
         border-radius: 12px; text-align: center; }
  #s0-big { position: absolute; left: 0; right: 0; top: 120px; text-align: center; }
  #s0-line1 { font-size: 64px; font-weight: 700; color: #E8E4DA; letter-spacing: 0.05em; }
  #s0-line2 { font-size: 88px; font-weight: 700; color: #D89A3A; font-family: 'JetBrains Mono', 'Noto Sans SC', monospace; margin-top: 12px; }
  #s0-line2 sub { font-size: 0.55em; }
  #s3-code { position: absolute; top: 210px; left: 50%; transform: translateX(-50%); width: 1120px;
             background: rgba(22,27,35,0.9); border: 2px solid rgba(154,163,178,0.25); border-radius: 14px;
             padding: 28px 44px; }
  .codeline { font: 30px 'JetBrains Mono', 'Noto Sans SC', monospace; color: #C9D4E4; line-height: 1.85; white-space: pre; }
  #s3-code-2, #s3-code-5 { color: #D89A3A; }
</style>
</head>
<body>
<div id="root" data-composition-id="dlab" data-start="0" data-width="1920" data-height="1080" data-duration="$TOTAL">
  <div id="bgfill"></div>
  <canvas id="maincv" class="clip" width="1920" height="1080" data-start="0" data-duration="$TOTAL" data-track-index="1"></canvas>
$SCENES
$CAPTIONS
$AUDIOS
</div>
<script>
(function () {
  var T = window.HF_TIMING, KC = window.HF_SCENES_CORE, KL = window.HF_SCENES_LATE;
  var cv = document.getElementById('maincv');
  var ctx = cv.getContext('2d');
  var order = ['s0','s1','s2','s3','s4','s5','s6','s7','s8','s9'];
  var draws = { s0: KC.drawS0, s1: KC.drawS1, s2: KC.drawS2, s3: KC.drawS3, s4: KC.drawS4,
                s5: KL.drawS5, s6: KL.drawS6, s7: KL.drawS7, s8: KL.drawS8, s9: KL.drawS9 };
  function drawAll(t) {
    ctx.clearRect(0, 0, 1920, 1080);
    for (var i = 0; i < order.length; i++) {
      var sid = order[i], S = T[sid];
      if (t >= S.start - 0.05 && t < S.start + S.dur) { draws[sid](ctx, t, T); break; }
    }
  }
  window.__timelines = window.__timelines || {};
  var tl = gsap.timeline({ paused: true });
  var proxy = { t: 0 };
  tl.to(proxy, { t: T.TOTAL, duration: T.TOTAL, ease: 'none',
    onUpdate: function () { drawAll(proxy.t); } }, 0);
  // scene chrome tweens (absolute times; never touch .clip elements themselves)
  order.forEach(function (sid) {
    var S = T[sid];
    var eb = document.getElementById(sid + '-eyebrow');
    if (eb) {
      tl.fromTo('#' + sid + '-eyebrow', { autoAlpha: 0, y: 16 }, { autoAlpha: 1, y: 0, duration: 0.7, ease: 'power2.out' }, S.start + 0.15);
      tl.fromTo('#' + sid + '-title', { autoAlpha: 0, y: 22 }, { autoAlpha: 1, y: 0, duration: 0.8, ease: 'power2.out' }, S.start + 0.3);
    }
  });
  // S0 big line at s0b
  tl.fromTo('#s0-line1', { autoAlpha: 0, y: 26 }, { autoAlpha: 1, y: 0, duration: 0.9, ease: 'power2.out' }, T.s0.segs[1].start + 0.2);
  tl.fromTo('#s0-line2', { autoAlpha: 0, scale: 0.92 }, { autoAlpha: 1, scale: 1, duration: 0.9, ease: 'power2.out' }, T.s0.segs[1].start + 1.0);
  // S3 code lines staggered during s3a
  for (var i = 0; i < 6; i++) {
    tl.fromTo('#s3-code-' + i, { autoAlpha: 0, x: -18 }, { autoAlpha: 1, x: 0, duration: 0.5, ease: 'power2.out' }, T.s3.segs[0].start + 0.6 + i * 1.1);
  }
  // hide the code block once density chart phase begins
  tl.to('#s3-code', { autoAlpha: 0, duration: 0.6 }, T.s3.segs[1].start + 0.2);
  window.__timelines['dlab'] = tl;
})();
</script>
</body>
</html>
'''
html = (HTML.replace('$TOTAL', str(TOTAL))
            .replace('$SCENES', '\n'.join(scene_els))
            .replace('$CAPTIONS', '\n'.join(caption_els))
            .replace('$AUDIOS', '\n'.join(audio_els)))
open(os.path.join(HERE, 'index.html'), 'w', encoding='utf-8').write(html)
print('index.html written', len(html) // 1024, 'KB')
