"""Illustrative image-space flight continuation, NOT calibrated aerodynamics.
Uses hand-set hang time, pixel/height and curvature heuristics. Does not integrate
physical drag or Magnus forces, and cannot measure carry from monocular pixels.
Optional recorded outcomes make this a reference visualization, not a prediction.
Excluded from all audited model inputs. See reports/gemini_model_audit.md."""
import numpy as np

def forecast(observations, ground_y=1100.0, curve_ft=None, carry_side_ft=None, height_ft=None, hang_time_s=None, carry_yd=None):
    if len(observations) < 4:
        return dict(status='unavailable', reason='Too few tracked points (minimum 4)')
    
    x0, y0 = observations[0]['x'], observations[0]['y']
    x_last, y_last = observations[-1]['x'], observations[-1]['y']
    t_start, t_last = observations[0]['playback_s'], observations[-1]['playback_s']
    span = max(0.015, t_last - t_start)
    
    dx = x_last - x0
    dy = y_last - y0
    vx = dx / span
    vy = dy / span
    
    climb_so_far = max(5.0, y0 - y_last)
    
    # Check for mishit grounder / worm-burner
    is_grounder = (dy > -60.0 and y_last >= ground_y - 35.0) or (height_ft is not None and height_ft <= 5.0)
    
    # 1. Autonomous estimation fallbacks if TrackMan is not used
    if hang_time_s is None or np.isnan(hang_time_s) or hang_time_s <= 0:
        if is_grounder:
            hang_time = 2.0
        else:
            # Scaled to launch rise: full 7-iron climb ~350-450 px maps to ~6.0 - 6.8s
            hang_time = float(np.clip(2.0 + (climb_so_far / 380.0) * 4.5, 2.0, 7.2))
    else:
        hang_time = float(hang_time_s)
        
    if height_ft is None or np.isnan(height_ft) or height_ft <= 0:
        if is_grounder:
            phys_height = 2.0
        else:
            phys_height = float(np.clip(15.0 + (climb_so_far / 380.0) * 60.0, 15.0, 110.0))
    else:
        phys_height = float(height_ft)
        
    if curve_ft is None or np.isnan(curve_ft):
        # Autonomous lateral curvature: estimate from initial launch angle dx/dy
        curve_val = float(np.clip((dx / max(10.0, abs(dy))) * 80.0, -90.0, 90.0))
    else:
        curve_val = float(curve_ft)
        
    # 2. Grounder trajectory
    if is_grounder:
        duration = float(np.clip(hang_time, 1.2, 2.8))
        dt_arr = np.linspace(0.0, duration, 35)
        pts = []
        for dt in dt_arr:
            persp = 1.0 / (1.0 + 0.8 * dt)
            cur_x = x_last + (vx * dt + 0.5 * (curve_val / 100.0) * 40.0 * (dt ** 2)) * persp
            pts.append(dict(
                playback_s=float(t_last + dt),
                x=float(cur_x),
                y=float(ground_y)
            ))
        return dict(
            status='flat_ground_continuation',
            points=pts,
            duration_s=float(duration),
            landing_x=float(pts[-1]['x']),
            landing_y=float(ground_y)
        )
        
    # 3. Full airborne flight:
    # Physical apex height in image coordinates:
    # A height of 70-80 ft corresponds to y ~ 600 - 750 (well up into the sky)
    scaled_h_px = min(500.0, max(180.0, phys_height * 4.6))
    y_apex = ground_y - scaled_h_px
    if y_apex >= y_last:
        y_apex = y_last - 50.0
        
    # Realistic timing:
    # Apex occurs ~48% of total flight time after impact
    t_apex_rel = max(0.4, hang_time * 0.48)
    t_land_rel = max(t_apex_rel + 0.5, hang_time)
    
    dt_last_rel = t_last - t_start
    rem_duration = max(0.8, t_land_rel - dt_last_rel)
    t_apex_from_last = max(0.3, t_apex_rel - dt_last_rel)
    t_fall = max(0.5, rem_duration - t_apex_from_last)
    
    # Lateral curvature acceleration in pixels/s^2 from curve_val (slice +, hook -)
    # Calibrated so curvature accumulates realistically over 5-6s downrange
    ax = (curve_val / 100.0) * 45.0
    
    dt_arr = np.linspace(0.0, rem_duration, 55)
    pts = []
    for dt in dt_arr:
        # Vertical motion: smooth rise to apex, smooth quadratic descent to flat ground horizon
        if dt <= t_apex_from_last:
            u = dt / t_apex_from_last
            cur_y = y_last - (y_last - y_apex) * (2.0 * u - u * u)
        else:
            v = (dt - t_apex_from_last) / t_fall
            cur_y = y_apex + (ground_y - y_apex) * (v * v)
        if cur_y > ground_y:
            cur_y = ground_y
            
        # Lateral motion: monotonic downrange position following initial velocity + aerodynamic curve
        persp = 1.0 / (1.0 + 0.35 * dt)
        cur_x = x_last + (vx * dt + 0.5 * ax * (dt ** 2)) * persp
        
        pts.append(dict(
            playback_s=float(t_last + dt),
            x=float(cur_x),
            y=float(cur_y)
        ))
        
    return dict(
        status='flat_ground_continuation',
        points=pts,
        duration_s=float(rem_duration),
        hang_time_s=float(hang_time),
        landing_x=float(pts[-1]['x']),
        landing_y=float(pts[-1]['y'])
    )
