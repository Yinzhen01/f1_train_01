"""Unfiltered temporal comparison: resize/label only, keep native video timing."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--case', action='append', required=True, help='ASCII-label=existing-video')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases = [entry.split('=', 1) for entry in a.case]
    if len(cases) < 2 or len({label for label, path in cases}) != len(cases):
        raise ValueError('At least two distinct cases required')
    for label, path in cases:
        if not re.fullmatch('[A-Za-z0-9_-]+', label) or not Path(path).is_file():
            raise ValueError('Invalid label or missing video')
    a.output.mkdir(parents=True, exist_ok=True)
    if any(a.output.iterdir()): raise ValueError('Refuse existing output files')
    results = []
    for label, start, duration in (('near5s', 4, 2), ('near29to30s', 28, 3), ('near48s', 47, 2)):
        target = a.output/(label+'.mp4')
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n']
        filters = []
        for index, (name, path) in enumerate(cases):
            command += ['-ss', str(start), '-t', str(duration), '-i', str(Path(path).resolve())]
            filters.append("[%d:v]scale=640:300,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text=%s:x=8:y=30:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6[v%d]" % (index, name, index))
        filters.append(''.join('[v%d]' % i for i in range(len(cases)))+'vstack=inputs=%d[out]' % len(cases))
        command += ['-filter_complex_threads', '1', '-filter_complex', ';'.join(filters), '-map', '[out]',
                    '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(target)]
        subprocess.run(command, check=True)
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(target),
                        '-vf', 'fps=15,scale=480:-1:flags=lanczos', str(a.output/(label+'.gif'))], check=True)
        results.append(dict(path=str(target.resolve()), start_s=start, duration_s=duration))
    (a.output/'manifest.json').write_text(json.dumps(dict(cases=cases, clips=results,
        scope='Resize, label, and cut only. No time warping/interpolation/smoothing; 50Hz videos represent recorded 100Hz trajectories.'), indent=2), encoding='utf-8')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
