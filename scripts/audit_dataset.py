"""Read-only source audit. Session links are filename-based candidates, never shot matches."""
import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def timing(values, scale=1.0, gap_multiple=1.5):
    t = np.asarray(values, dtype=float) * scale
    d = np.diff(t)
    positive = d[d > 0]
    median = float(np.median(positive)) if len(positive) else None
    return dict(first_s=float(t[0]), last_s=float(t[-1]), duration_s=float(t[-1]-t[0]),
                median_dt_s=median, median_hz=1/median if median else None,
                mean_hz=float((len(t)-1)/(t[-1]-t[0])) if t[-1]>t[0] else None,
                max_gap_s=float(d.max()) if len(d) else None,
                nonpositive_steps=int((d<=0).sum()),
                gaps_over_threshold=int((d>gap_multiple*median).sum()) if median else None)


def run(root, config):
    out = root/'data/manifests'
    out.mkdir(parents=True, exist_ok=True)
    cache = root/'outputs/audit/cache'
    cache.mkdir(parents=True, exist_ok=True)
    inventory = []
    for folder in config['raw_directories']:
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.name != '.DS_Store':
                inventory.append(p)
    inventory.append(root/config['workbook'])
    # Hash all bytes once; cache hashes by relative path, byte size and mtime.
    cachefile = cache/'hashes.json'
    old = json.loads(cachefile.read_text()) if cachefile.exists() else {}
    hashes, files = {}, []
    for p in inventory:
        rel = str(p.relative_to(root))
        key = f'{rel}|{p.stat().st_size}|{p.stat().st_mtime_ns}'
        hashes[key] = old.get(key) or digest(p)
        files.append(dict(path=rel, bytes=p.stat().st_size, sha256=hashes[key]))
    cachefile.write_text(json.dumps(hashes, indent=2))
    pd.DataFrame(files).to_csv(out/'file_inventory.csv', index=False)
    duplicates = defaultdict(list)
    for f in files:
        duplicates[f['sha256']].append(f['path'])

    wb = openpyxl.load_workbook(root/config['workbook'], read_only=True, data_only=True)
    sheets = {s.title:list(s.values) for s in wb}
    wb.close()
    (out/'workbook_values.json').write_text(json.dumps(sheets, indent=2, default=str))
    rows = [dict(zip(sheets['Shots'][0], row)) for row in sheets['Shots'][1:] if row[1]]
    manifest = []
    for i, row in enumerate(rows, 2):
        video, club, watch = config['sessions'][row['Session']]
        manifest.append(dict(shot_id=row['Shot ID'], physical_shot_verified=False,
            workbook_row=i, golfer=row['Golfer'], session=row['Session'],
            shot_in_session=row['Shot in session'], trackman_status=row['TrackMan status'],
            label_review_status=row['Review status'], video_candidate=video,
            club_candidate=club+'/imu.csv', watch_candidate=watch+'/WristMotion.csv',
            association_status='order_rule_user_confirmed_windows_pending' if config.get('order_confirmed_by_user') else 'provisional_session_only',
            association_confidence='order_confirmed_events_unverified' if config.get('order_confirmed_by_user') else 'low',
            evidence='User confirms nth actual strike corresponds to nth session row; no practice swings; club may stop early' if config.get('order_confirmed_by_user') else 'matching session names; workbook says watch/photo alignment inferred',
            video_impact_s=None, club_impact_s=None, watch_impact_s=None,
            training_eligible=False))
    pd.DataFrame(manifest).to_csv(out/'shot_manifest.csv', index=False)
    pd.DataFrame(rows).to_csv(out/'trackman_labels.csv', index=False)

    sensors = []
    for p in inventory:
        if p.suffix.lower() != '.csv':
            continue
        rel = str(p.relative_to(root))
        try:
            df = pd.read_csv(p)
            entry = dict(path=rel, rows=len(df), columns=list(df.columns),
                         missing={k:int(v) for k,v in df.isna().sum().items() if v},
                         exact_duplicate_rows=int(df.duplicated().sum()))
            if len(df) and ('t_ns' in df or 'seconds_elapsed' in df):
                col = 't_ns' if 't_ns' in df else 'seconds_elapsed'
                entry['timing'] = timing(df[col], 1e-9 if col=='t_ns' else 1, config['gap_multiple'])
            numeric = df.select_dtypes(include='number')
            entry['ranges'] = {c:[float(numeric[c].min()),float(numeric[c].max())]
                               for c in numeric if numeric[c].notna().any()}
            for flag in ['near_accel_limit','near_gyro_limit']:
                if flag in df:
                    flagged = df.loc[df[flag] != 0]
                    entry[flag+'_rows'] = len(flagged)
                    entry[flag+'_first_s'] = float(flagged.t_ns.iloc[0]*1e-9) if len(flagged) else None
            sensors.append(entry)
        except Exception as exc:
            sensors.append(dict(path=rel, error=str(exc)))
    (out/'sensor_audit.json').write_text(json.dumps(sensors, indent=2, allow_nan=False))

    videos = []
    for p in inventory:
        if p.suffix.lower() not in ['.mov','.mp4']:
            continue
        rel = str(p.relative_to(root))
        ck = hashlib.sha256((rel+str(p.stat().st_size)+str(p.stat().st_mtime_ns)).encode()).hexdigest()
        metadata = cache/(ck+'.json')
        if not metadata.exists():
            result = subprocess.run(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(p)], capture_output=True, text=True)
            if result.returncode:
                videos.append(dict(path=rel, error=result.stderr.strip().replace(str(root)+'/',''), returncode=result.returncode))
                continue
            metadata.write_text(result.stdout)
        info = json.loads(metadata.read_text())
        if 'format' in info:
            info['format']['filename'] = rel
        entry = dict(path=rel, metadata=info)
        if p.suffix.lower()=='.mov':
            ptsfile = cache/(ck+'_packets.json')
            if not ptsfile.exists():
                result = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_packets',
                    '-show_entries','packet=pts_time,duration_time','-of','json',str(p)],capture_output=True,text=True,check=True)
                ptsfile.write_text(result.stdout)
            packets = json.loads(ptsfile.read_text())['packets']
            pts = sorted(float(x['pts_time']) for x in packets if 'pts_time' in x)
            entry['packet_count'] = len(packets)
            entry['presentation_timing'] = timing(pts)
            entry['packet_duration_counts'] = dict(Counter(x.get('duration_time','missing') for x in packets))
        videos.append(entry)
    (out/'media_audit.json').write_text(json.dumps(videos, indent=2))
    summary = dict(file_count=len(files), bytes=sum(f['bytes'] for f in files),
        shot_candidates=len(rows), sessions=dict(Counter(r['Session'] for r in rows)),
        trackman_status=dict(Counter(r['TrackMan status'] for r in rows)),
        duplicate_file_groups=[v for v in duplicates.values() if len(v)>1],
        missing_source_photos=sorted(set(r['Source photo'] for r in rows if r['Source photo']
            and not (root/'source_photos'/r['Source photo']).exists())),
        sensor_errors=[s for s in sensors if 'error' in s],
        label_missing_counts={k:sum(r.get(k) is None for r in rows) for k in rows[0]},
        workbook_sha256=next(f['sha256'] for f in files if f['path']==config['workbook']))
    (out/'audit_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['missing_source_photos','duplicate_file_groups']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path('configs/audit.json'))
    args = parser.parse_args()
    run(args.root, json.loads((args.root/args.config).read_text()))
