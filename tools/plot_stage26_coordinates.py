"""Plot recorded world-coordinate estimates on the original CCTV frame."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.visualize_detection_eval import project, draw_cross, RED, CYAN

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump-dir', type=Path, required=True,
                        help='External CCTV dump with meta.json and round_26 frames; not bundled')
    parser.add_argument('--evaluation', type=Path, required=True,
                        help='External evaluation JSON containing round_26')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/stage26_gt_pred_cctv_debug_style.png')
    args = parser.parse_args()
    if not (args.dump_dir / 'meta.json').is_file():
        parser.error('The supplied CCTV dump must contain meta.json; the dataset is not bundled.')
    source = args.evaluation
    payload = json.loads(source.read_text(encoding='utf-8'))
    record = next(r for r in payload['rounds_detail'] if r['dir'] == 'round_26')
    dump = args.dump_dir
    meta = json.loads((dump / 'meta.json').read_text(encoding='utf-8'))
    camera = record['submitted']['camera_id']
    assert camera == record['expected']['camera_id']
    frame = dump / record['dir'] / f'{camera}.png'
    pixels = {k: project(meta, camera, record[k]['target_coord'])
              for k in ('expected', 'submitted')}
    assert all(p is not None and np.isfinite(p).all() for p in pixels.values())
    assert np.allclose(pixels['submitted'], record['vision_trace']['pose_refinement']['pixel'])
    original = Image.open(frame).convert('RGB')
    draw = ImageDraw.Draw(original)
    draw_cross(draw, pixels['expected'], RED, 'GT')
    draw_cross(draw, pixels['submitted'], CYAN, 'PRED')
    img = np.asarray(original)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(-.5, w-.5)
    ax.set_ylim(h-.5, -.5)
    ax.axis('off')
    ax.text(.015, .975, f'Stage 26  |  {camera}  |  Error: {record["target_err"]:.3f} m',
            transform=ax.transAxes, va='top', color='white', fontsize=14,
            bbox=dict(facecolor='black', alpha=.75, edgecolor='none', pad=7))
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)
    details = {'evaluation': source.name, 'frame': (Path(record['dir']) / frame.name).as_posix(), 'camera': camera,
               'target_error_m': record['target_err'],
               'world_coordinates': {k: record[k]['target_coord'] for k in pixels},
               'pixel_coordinates': {k: list(map(float, p)) for k, p in pixels.items()}}
    output.with_suffix('.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(json.dumps(details, indent=2))
    print(output)


if __name__ == "__main__":
    main()
