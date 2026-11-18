# Stable 3D range avatar — user-directed display update

Implemented the requested shift from the flat pose figure to an articulated 3D golfer in a stylized driving range. The scene contains hitting mats, bay partitions, a ball basket, target greens/flags, distant trees and nets, lighting and shadows. The character has a solid torso, trousers, shoes, head/cap, arms and both legs. A club shaft/grip/iron head stays attached to the common hand grip.

The camera can orbit, remain close to the golfer, switch to a flight view, or automatically pull back after the provisional departure time. A visible ball and trailing line illustrate flight. An experimental tracer is also drawn directly over the native video using saved 2D automatic track points, without extrapolating that overlay beyond observed proposals.

## What is inferred

The display rig temporally fills uncertain/missing joints, smooths motion, constrains leg segment lengths, and completes the elbows around a shared golf grip. Both legs remain opaque; visibility confidence no longer makes limbs fade in/out. These are display-only guesses supported by surrounding pose frames and plausible proportions, not newly measured joints. The original pose caches, feature tables and predictions remain unchanged.

Club orientation is a phase-driven illustration connected to the hands; the system does not yet measure the full club orientation. The 3D flight is a compressed schematic arc scaled to the carry estimate when available, with approximate lateral illustration when reviewed points exist. It is not calibrated 3D reconstruction, a physics-based carry estimate, measured spin/curve, or evidence of where the actual ball landed. Unavailable carry uses a generic illustrative arc rather than creating a numerical prediction.

The native-video overlay is distinct: it uses the saved automatic image-plane trajectory. Those proposals still sometimes follow shafts or background and remain excluded from predictive features. The UI labels both limitations.

## Verification

- 24 unit tests pass, including missing-pose completion, stable leg lengths, shared grip and source-pose immutability.
- Scene-graph math exercised on all 7,980 frames from 76 swings: finite transforms, both legs visible throughout and a consistent common grip.
- `scripts/verify_range_scene.js` uses the actual Three.js scene objects with a geometry-only renderer stub. This is not a browser, WebGL render or interaction test.
- Software projection previews of the actual scene geometry were inspected; the flight camera was widened to retain the golfer while showing the tracer. Preview: `outputs/range_scene_preview/range_preview.png`. These are explicitly labeled geometry previews, not app screenshots.
- Existing 76 pose clips and 19,938 native frames retain their media/data checks. Updated local research page responds HTTP 200.
- Browser access remains unavailable after the user's reset. No native-app selection was retried. Browser/WebGL visual and interaction testing is still pending; user assistance with browser access was requested asynchronously.

## Dependencies and reproduction

Three.js 0.160.1 is vendored from the official npm package, with SHA-256 provenance and its MIT license. It is embedded locally so the HTML can open without a CDN, network connection or module server. Library documentation: https://threejs.org/docs/ . The classic build is pinned for local-file compatibility; its deprecation notice is preserved.

```
.venv/bin/python scripts/avatar_motion.py
.venv/bin/python scripts/build_research_viewer.py
node scripts/verify_range_scene.js
.venv/bin/python scripts/render_range_geometry_preview.py
.venv/bin/python scripts/verify_research_delivery.py
.venv/bin/python -m unittest discover -s tests -q
```

New-video analyses also use the same completed display rig. Completely missing pose still causes prediction abstention and hides the character instead of inventing a swing. Restore browser access for final visual validation; prioritize actual club/ball measurement and independent pose review separately from this display layer.
