"""Reproduce production candidate geometry; sample disjoint rejected RF bands."""
import csv, hashlib, json, os, random, sys
from pathlib import Path
# [REDACTED for publication: HPC username, private lab repository name, and a collaborator's
#  name removed. Set SCRATCH_ROOT and TILING_CODE_DIR.]
# ROOT = Path('/scratch/<REDACTED_USER>')
# CODE = ROOT / '<REDACTED_LAB_REPOSITORY>/<REDACTED_COLLABORATOR>_tiling'
ROOT = Path(os.environ.get('SCRATCH_ROOT', '/path/to/scratch'))
CODE = Path(os.environ.get('TILING_CODE_DIR', '/path/to/tiling-code'))
sys.path.insert(0, str(CODE))
OUT = Path(__file__).resolve().parent
BANDS = [('yield_rf070', .70, .75), ('yield_rf065', .65, .70), ('yield_rf060', .60, .65)]
SEED = 20260921

def write_csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def select():
    from diagnose_empty_scans import SUSPECT, CONTROL
    with open(ROOT / 'mycorrhisee_tiling_output/_logs/diagnostics/tile_yield_all_scans.csv') as f:
        audit = list(csv.DictReader(f))
    bypath = {r['rel_input_path']: r for r in audit}
    chosen = [(p, 'zero_yield') for p in SUSPECT]
    rng = random.Random(SEED)
    for block in ('Block5', 'Block11', 'Block12'):
        pool = sorted(r['rel_input_path'] for r in audit if r['block'] == block and r['low_yield_le5pct'] == '1' and r['rel_input_path'] not in SUSPECT)
        chosen.extend((p, 'low_yield') for p in rng.sample(pool, 2))
    chosen.extend((p, 'control') for p in CONTROL)
    rows = [dict(bypath[p], sample_role=role) for p, role in chosen]
    write_csv(OUT / 'selected_scans.csv', rows)
    return rows

def main():
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw
    from aicsimageio import AICSImage
    from diagnose_empty_scans import COORDS_1152_576, PADDING
    from spline_guided_tiling_utils_v2 import preprocess_image, find_largest_contour_with_area, find_squares, warp_and_crop
    cv2.setNumThreads(2)
    rows = select(); summaries = []; manifest = []; errors = []
    (OUT / 'tiles').mkdir(exist_ok=True); (OUT / 'previews').mkdir(exist_ok=True)
    for idx, row in enumerate(rows):
        rel = row['rel_input_path']; name = Path(rel).stem
        uid = hashlib.sha256(rel.encode()).hexdigest()[:8]
        rng = random.Random(SEED + idx)
        counts = {b[0]: 0 for b in BANDS}; samples = {b[0]: [] for b in BANDS}
        candidates = baseline = completed = nonrgb = 0
        print(f'SCAN {idx+1}/{len(rows)} {rel}', flush=True)
        czi = AICSImage(str(ROOT / 'mycorrhisee/MycorrhiSEE' / rel))
        indices = sorted(set(np.linspace(0, len(czi.scenes)-1, min(6, len(czi.scenes)), dtype=int).tolist()))
        for si in indices:
            scene = czi.scenes[si]
            try:
                czi.set_scene(scene); im = czi.get_image_data('YXS', T=0, C=0, Z=0)
                if im.ndim != 3 or im.shape[-1] != 3:
                    nonrgb += 1; continue
                im[(im == [0,0,0]).all(axis=-1)] = 255
                halves = preprocess_image(im, threshold=175, gray=False, rotate=True, half=True, half_rotation=False)
                for hi, half in enumerate(halves, 1):
                    if half is None: continue
                    half[(half == [0,0,0]).all(axis=-1)] = 255
                    padded = cv2.copyMakeBorder(half, PADDING, PADDING, PADDING, PADDING, cv2.BORDER_CONSTANT, value=[255]*3)
                    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
                    contour, x, y, _ = find_largest_contour_with_area(gray, 175)
                    if contour is None: continue
                    try:
                        xs, ys, _ = find_squares(x, y, x.min(), x.max(), side_length=1152, step_size=1152, move=True, ratio=.33)
                    except ValueError as e:
                        errors.append(dict(rel_input_path=rel, scene=scene, error=f'half {hi} skipped as in production: {e!r}'))
                        continue
                    for pi, (xc, yc) in enumerate(zip(xs, ys), 1):
                        try:
                            patch = warp_and_crop(padded, xc, yc, side_length=1152)
                        except (ValueError, cv2.error) as e:
                            errors.append(dict(rel_input_path=rel, scene=scene, error=f'half {hi} patch {pi} skipped: {e!r}'))
                            continue
                        for ti, (y1,y2,x1,x2) in enumerate(COORDS_1152_576, 1):
                            tile = patch[y1:y2,x1:x2]
                            g = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                            rf = float(np.mean(g < 175)); candidates += 1; baseline += int(rf >= .75)
                            for exp, lo, upper in BANDS:
                                if lo <= rf < upper:
                                    counts[exp] += 1
                                    # Uniform reservoir within each scan and exclusive RF band.
                                    n = counts[exp]; slot = n-1 if n <= 12 else rng.randrange(n)
                                    if slot < 12:
                                        filename = f'{exp}__{row["block"]}__{name}__{uid}__{scene}_half{hi}_patch{pi}_tile{ti}__RF{rf:.4f}.png'
                                        meta = dict(experiment=exp, rf_lower=lo, rf_upper_exclusive=upper, rel_input_path=rel, sample_role=row['sample_role'], block=row['block'], scene=scene, half=hi, patch=pi, tile=ti, root_fraction=rf, fraction_near_white=float(np.mean(g >= 245)), filename=filename, visual_review='pending')
                                        item = (meta, tile.copy())
                                        if n <= 12: samples[exp].append(item)
                                        else: samples[exp][slot] = item
                completed += 1
                print(f'  {scene}: cumulative candidates={candidates}', flush=True)
            except Exception as e:
                errors.append(dict(rel_input_path=rel, scene=scene, error=repr(e)))
                print(f'ERROR {scene}: {e!r}', flush=True)
        added = 0
        for exp, lo, upper in BANDS:
            added += counts[exp]
            summaries.append(dict(rel_input_path=rel, sample_role=row['sample_role'], block=row['block'], experiment=exp, rf_cutoff=lo, selected_scenes=len(indices), completed_scenes=completed, skipped_nonrgb=nonrgb, total_scenes=len(czi.scenes), scope='sampled_scenes_only', candidate_tiles=candidates, baseline_kept_rf075=baseline, baseline_yield=baseline/candidates if candidates else '', experiment_kept=baseline+added, experiment_yield=(baseline+added)/candidates if candidates else '', additional_tiles_vs_rf075=added, incremental_tiles_in_band=counts[exp], saved_sample_tiles=len(samples[exp]), historical_full_scan_yield=row['keep_rate'], scene_error_count=sum(e['rel_input_path']==rel for e in errors)))
            sheet = Image.new('RGB', (4*256, 3*286), 'white'); draw = ImageDraw.Draw(sheet)
            for j, (meta, tile) in enumerate(samples[exp]):
                path = OUT / 'tiles' / meta['filename']
                if not cv2.imwrite(str(path), tile): raise IOError(path)
                manifest.append(meta)
                thumb = Image.open(path).convert('RGB'); thumb.thumbnail((256,256))
                px, py = (j%4)*256, (j//4)*286
                sheet.paste(thumb, (px,py)); draw.text((px+3,py+258), f'{meta["scene"]} RF={meta["root_fraction"]:.4f}', fill='black')
            if samples[exp]: sheet.save(OUT / 'previews' / f'{exp}__{name}__{uid}.jpg')
        write_csv(OUT / 'yield_by_scan_experiment.csv', summaries)
        write_csv(OUT / 'sampled_tiles.csv', manifest)
        (OUT / 'errors.json').write_text(json.dumps(errors, indent=2))
    (OUT / 'COMPLETE.json').write_text(json.dumps(dict(scans=len(rows), saved_tiles=len(manifest), seed=SEED, errors=len(errors)), indent=2))
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    if '--select-only' in sys.argv: select()
    else: main()
