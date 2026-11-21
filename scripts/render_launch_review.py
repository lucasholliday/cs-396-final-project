"""Render visual proof sheets showing tracked pre-impact clubhead and post-impact ball flight."""
import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def render_shot_review(sid, out_path):
    track_path = ROOT / f'data/processed/ball_launch_tracks/{sid}.json'
    native_meta_path = ROOT / f'outputs/native_review/{sid}.json'
    native_video_path = ROOT / f'outputs/native_review/{sid}.mp4'
    if not (track_path.exists() and native_meta_path.exists() and native_video_path.exists()):
        return False
        
    data = json.loads(track_path.read_text())
    meta = json.loads(native_meta_path.read_text())
    ball_obs = data.get('observations', [])
    club_obs = data.get('club_track', [])
    seed = data.get('seed')
    metrics = data.get('metrics', {})
    
    cap = cv2.VideoCapture(str(native_video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        return False
        
    # Pick 4 representative frames:
    # 1: Club approaching impact
    # 2: At impact / seed disappearance
    # 3: Ball early departure
    # 4: Full flight + tracer
    sf = seed['frame'] if seed else 114
    target_frames = [max(0, sf - 2), sf, min(len(frames) - 1, sf + 2), min(len(frames) - 1, sf + 6)]
    
    panels = []
    # Crop tightly to the hitting zone
    # Native review video is 780x1100, origin is (300, 650) in 1080x1920
    for f_idx in target_frames:
        frame = frames[f_idx].copy()
        t = meta['source_pts'][f_idx] if f_idx < len(meta['source_pts']) else 0.0
        
        # Draw seed location (white dashed ring)
        if seed:
            sx, sy = int(seed['x'] - 300), int(seed['y'] - 650)
            cv2.circle(frame, (sx, sy), 8, (255, 255, 255), 1)
            
        # Draw clubhead track (cyan / teal)
        for i in range(1, len(club_obs)):
            if club_obs[i]['frame'] <= f_idx:
                p0 = (int(club_obs[i-1]['x'] - 300), int(club_obs[i-1]['y'] - 650))
                p1 = (int(club_obs[i]['x'] - 300), int(club_obs[i]['y'] - 650))
                cv2.line(frame, p0, p1, (220, 220, 60), 2) # BGR cyan
                cv2.circle(frame, p1, 5, (240, 240, 70), -1)
                
        # Draw ball track (gold / orange)
        active_ball = [b for b in ball_obs if b['frame'] <= f_idx]
        for i in range(1, len(active_ball)):
            p0 = (int(active_ball[i-1]['x'] - 300), int(active_ball[i-1]['y'] - 650))
            p1 = (int(active_ball[i]['x'] - 300), int(active_ball[i]['y'] - 650))
            cv2.line(frame, p0, p1, (40, 180, 255), 3) # BGR gold
            cv2.circle(frame, p1, 4, (60, 220, 255), -1)
        if active_ball:
            latest = (int(active_ball[-1]['x'] - 300), int(active_ball[-1]['y'] - 650))
            cv2.circle(frame, latest, 6, (0, 255, 255), -1)
            cv2.circle(frame, latest, 8, (255, 255, 255), 1)
            
        # Crop region of interest: around mat and flight path
        # Mat is near bottom center
        h, w = frame.shape[:2]
        crop = frame[max(0, h - 600):h, max(0, w // 2 - 250):min(w, w // 2 + 250)]
        crop = cv2.resize(crop, (280, 360))
        
        cv2.putText(crop, f'f={f_idx}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        panels.append(crop)
        
    strip = np.hstack(panels)
    
    # Add title banner
    banner = np.zeros((55, strip.shape[1], 3), dtype=np.uint8)
    banner[:] = (20, 30, 35)
    ang = metrics.get('ball_angle_deg', 0.0)
    spd = metrics.get('ball_speed_px_s', 0.0)
    cspd = metrics.get('club_speed_px_s', 0.0)
    cv2.putText(banner, f'{sid}: Tracked Launch Vector (Cyan=Clubhead, Gold=Ball)', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(banner, f'Ball Angle: {ang:+.1f} deg | Ball Speed: {spd:.0f} px/s | Clubhead: {cspd:.0f} px/s', (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 230, 160), 1)
    
    combined = np.vstack([banner, strip])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)
    return True


if __name__ == '__main__':
    out_dir = ROOT / 'outputs/launch_review_sheets'
    for sid in ['lucas_1_02', 'lucas_1_09', 'lucas_1_10', 'lucas_1_12']:
        ok = render_shot_review(sid, out_dir / f'{sid}_launch.jpg')
        print(f'{sid}: rendered={ok}')
