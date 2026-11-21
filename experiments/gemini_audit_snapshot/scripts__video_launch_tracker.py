"""Physics-constrained ballistic ball tracking and pre-impact clubhead speed extraction.
Eliminates clubhead follow-through confusion and extracts pure launch vector features.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from link_ball_v6 import follow


def extract_shot_launch(sid, cfg=None):
    if cfg is None:
        cfg = json.loads((ROOT / 'configs/video_launch_tracking.json').read_text())
        
    ball_path = ROOT / f'data/processed/ball_v7/{sid}.json'
    if not ball_path.exists():
        return None
    d = json.loads(ball_path.read_text())
    seeds = d.get('seeds', [])
    if not seeds:
        return None
    seed = seeds[0]
    
    # 1. Extract candidate ball tracks departing from seed
    candidate_tracks = []
    for f in d['frames']:
        elapsed = f['playback_s'] - seed['playback_s']
        if not (cfg['seed_min_departure_elapsed_s'] <= elapsed <= cfg['seed_max_departure_elapsed_s']):
            continue
        for c in f['candidates']:
            if c.get('polarity') != 1 or c['area'] < cfg['initial_candidate_min_area']:
                continue
            
            # Initial departure point must be close to seed and not approaching from far left
            dist = np.hypot(c['x'] - seed['x'], c['y'] - seed['y'])
            if not (cfg['initial_candidate_min_dist_px'] < dist < cfg['initial_candidate_max_dist_px']):
                continue
            if c['y'] >= seed['y']:
                continue  # Must be moving upward
            if c['x'] < seed['x'] - cfg['initial_candidate_left_margin_px']:
                continue  # Reject clubhead approaching from the inside/left
            
            tr = follow(d['frames'], f['frame'], c, seed)
            if len(tr) >= cfg['min_track_length'] and (tr[0]['y'] - tr[-1]['y']) >= cfg['min_vertical_rise_px']:
                pts = np.array([[p['x'], p['y'], p['playback_s']] for p in tr])
                x, y, t = pts[:, 0], pts[:, 1], pts[:, 2]
                dt = t[-1] - t[0]
                if dt < cfg['min_track_duration_s']:
                    continue
                dx_tot = x[-1] - x[0]
                dy_tot = y[0] - y[-1]
                angle_deg = float(np.degrees(np.arctan2(dx_tot, dy_tot)))
                speed = float(np.hypot(dx_tot, dy_tot) / dt)
                
                # Check if this is an obvious clubhead follow-through sweeping left
                is_club = (dx_tot < cfg['club_follow_through_dx_threshold_px'] and 
                           angle_deg < cfg['club_follow_through_angle_threshold_deg'])
                
                # Straightness metric: scatter around linear regression ray
                res_x = float(np.abs(x - np.polyval(np.polyfit(t - t[0], x, 1), t - t[0])).mean())
                
                score = (
                    min(len(tr), 12) * 2.0 
                    + (speed / 1000.0) * 1.5 
                    - res_x * 0.5 
                    - (100.0 if is_club else 0.0)
                )
                candidate_tracks.append({
                    'score': score,
                    'track': tr,
                    'angle_deg': angle_deg,
                    'speed_px_s': speed,
                    'dx_px': float(dx_tot),
                    'dy_px': float(dy_tot),
                    'dt_s': float(dt),
                    'is_club': is_club,
                    'res_x': res_x
                })
                
    if not candidate_tracks:
        return None
        
    candidate_tracks.sort(key=lambda q: -q['score'])
    best = candidate_tracks[0]
    
    # 2. Extract pre-impact clubhead speed
    sf = seed['frame']
    pre_cands = []
    for f in d['frames']:
        if sf - cfg['clubhead_pre_impact_frames'] <= f['frame'] <= sf:
            cands = [c for c in f['candidates'] if abs(c['x'] - seed['x']) < cfg['clubhead_search_radius_x_px'] and 
                     abs(c['y'] - seed['y']) < cfg['clubhead_search_radius_y_px']]
            if cands:
                pre_cands.append((f['frame'], f['playback_s'], cands[0]))
                
    club_speed_px = np.nan
    club_track = []
    if len(pre_cands) >= 2:
        dt_club = pre_cands[-1][1] - pre_cands[0][1]
        if dt_club > 0:
            c0, c1 = pre_cands[0][2], pre_cands[-1][2]
            club_speed_px = float(np.hypot(c1['x'] - c0['x'], c1['y'] - c0['y']) / dt_club)
            club_track = [{'frame': f, 'playback_s': t, 'x': c['x'], 'y': c['y']} for f, t, c in pre_cands]
            
    smash_proxy = (best['speed_px_s'] / club_speed_px) if (club_speed_px and club_speed_px > 0) else np.nan
    
    return {
        'shot_id': sid,
        'seed': seed,
        'ball_track': best['track'],
        'ball_len': len(best['track']),
        'ball_angle_deg': best['angle_deg'],
        'ball_speed_px_s': best['speed_px_s'],
        'ball_rise_px': best['dy_px'],
        'ball_lateral_dx_px': best['dx_px'],
        'ball_duration_s': best['dt_s'],
        'ball_residual_x': best['res_x'],
        'club_speed_px_s': club_speed_px,
        'club_track': club_track,
        'smash_factor_proxy': smash_proxy,
        'mishit_push_flag': float(best['angle_deg'] > 15.0),
        'abs_angle_deg': float(abs(best['angle_deg']))
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    cfg = json.loads((ROOT / 'configs/video_launch_tracking.json').read_text())
    m = pd.read_csv(ROOT / 'data/manifests/shot_manifest.csv')
    
    out_dir = ROOT / 'data/processed/ball_launch_tracks'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rows = []
    print(f'Extracting physics-gated launch features for {len(m)} shots...')
    for row in m.itertuples():
        res = extract_shot_launch(row.shot_id, cfg)
        if res is not None:
            # Save individual track json
            track_file = out_dir / f'{row.shot_id}.json'
            payload = {
                'shot_id': row.shot_id,
                'seed': res['seed'],
                'observations': res['ball_track'],
                'club_track': res['club_track'],
                'metrics': {
                    'ball_angle_deg': res['ball_angle_deg'],
                    'ball_speed_px_s': res['ball_speed_px_s'],
                    'ball_rise_px': res['ball_rise_px'],
                    'ball_lateral_dx_px': res['ball_lateral_dx_px'],
                    'ball_duration_s': res['ball_duration_s'],
                    'club_speed_px_s': res['club_speed_px_s'],
                    'smash_factor_proxy': res['smash_factor_proxy'],
                    'mishit_push_flag': res['mishit_push_flag']
                }
            }
            track_file.write_text(json.dumps(payload, indent=2))
            
            # Row for features CSV
            row_dict = {
                'shot_id': row.shot_id,
                'golfer': row.golfer,
                'session': row.session,
                'ball_tracked': 1,
                'ball_len': res['ball_len'],
                'ball_angle_deg': res['ball_angle_deg'],
                'ball_speed_px_s': res['ball_speed_px_s'],
                'ball_rise_px': res['ball_rise_px'],
                'ball_lateral_dx_px': res['ball_lateral_dx_px'],
                'ball_duration_s': res['ball_duration_s'],
                'ball_residual_x': res['ball_residual_x'],
                'club_speed_px_s': res['club_speed_px_s'],
                'smash_factor_proxy': res['smash_factor_proxy'],
                'mishit_push_flag': res['mishit_push_flag'],
                'abs_angle_deg': res['abs_angle_deg']
            }
            rows.append(row_dict)
        else:
            rows.append({
                'shot_id': row.shot_id,
                'golfer': row.golfer,
                'session': row.session,
                'ball_tracked': 0,
                'ball_len': 0,
                'ball_angle_deg': np.nan,
                'ball_speed_px_s': np.nan,
                'ball_rise_px': np.nan,
                'ball_lateral_dx_px': np.nan,
                'ball_duration_s': np.nan,
                'ball_residual_x': np.nan,
                'club_speed_px_s': np.nan,
                'smash_factor_proxy': np.nan,
                'mishit_push_flag': np.nan,
                'abs_angle_deg': np.nan
            })
            
    df_features = pd.DataFrame(rows)
    df_features.to_csv(ROOT / 'data/processed/video_launch_features.csv', index=False)
    print(f'Saved {len(rows)} feature rows to data/processed/video_launch_features.csv')
    print(f'Successfully tracked: {df_features.ball_tracked.sum()} / {len(m)} shots')


if __name__ == '__main__':
    main()
