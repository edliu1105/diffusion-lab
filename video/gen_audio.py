"""Generate per-segment narration mp3s from SCRIPT.md via edge-tts + duration manifest."""
import asyncio, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'assets', 'vo')
os.makedirs(OUT, exist_ok=True)
VOICE = 'zh-CN-YunxiNeural'
RATE = '+2%'

def parse_script():
    segs = []
    for line in open(os.path.join(HERE, 'SCRIPT.md'), encoding='utf-8'):
        m = re.match(r'^- (s\w+) \| (.+)$', line.strip())
        if m:
            segs.append((m.group(1), m.group(2).strip()))
    return segs

async def synth(seg_id, text):
    import edge_tts
    path = os.path.join(OUT, f'{seg_id}.mp3')
    for attempt in range(3):
        try:
            tts = edge_tts.Communicate(text, VOICE, rate=RATE)
            await tts.save(path)
            if os.path.getsize(path) > 1000:
                return path
        except Exception as e:
            print(f'  {seg_id} attempt {attempt}: {e}')
            await asyncio.sleep(2)
    raise RuntimeError(f'TTS failed for {seg_id}')

def dur(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                        '-of', 'csv=p=0', path], capture_output=True, text=True)
    return float(r.stdout.strip())

async def main():
    segs = parse_script()
    print(f'{len(segs)} segments')
    manifest = {}
    for sid, text in segs:
        p = os.path.join(OUT, f'{sid}.mp3')
        if not (os.path.exists(p) and os.path.getsize(p) > 1000):
            await synth(sid, text)
        d = dur(p)
        manifest[sid] = {'dur': round(d, 3), 'chars': len(text)}
        print(f'  {sid}: {d:.2f}s ({len(text)} chars)')
    json.dump(manifest, open(os.path.join(OUT, 'manifest.json'), 'w'), indent=1)
    total = sum(v['dur'] for v in manifest.values())
    print(f'total narration: {total/60:.1f} min')

if __name__ == '__main__':
    asyncio.run(main())
