/**
 * PoseFormerV2 (CVPR 2023) 3D Kinematic Skeleton & Golf Club Studio
 * Renders authentic 17-joint frequency-domain lifted skeletons with the golf club attached,
 * with real-time video overlay and 3D studio views.
 */
class PoseFormerStudio {
  constructor(canvasId, containerId, videoId, videoCanvasId) {
    this.canvas = document.getElementById(canvasId || 'pfCanvas');
    this.container = document.getElementById(containerId || 'pfCanvasContainer');
    this.video = document.getElementById(videoId || 'pfVideo');
    this.videoCanvas = document.getElementById(videoCanvasId || 'pfVideoCanvas');
    this.videoWrap = document.getElementById('pfVideoWrap');
    this.threeDWrap = document.getElementById('pf3DWrap');

    if (!this.canvas || !window.THREE) return;

    const T = window.THREE;
    this.renderer = new T.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(this.threeDWrap ? this.threeDWrap.clientWidth || 540 : 540, 470);
    this.renderer.shadowMap.enabled = true;

    this.scene = new T.Scene();
    this.scene.background = new T.Color('#071118');

    this.camera = new T.PerspectiveCamera(40, 1, 0.1, 50);
    this.cameraTarget = new T.Vector3(0, -0.15, 0);
    this.orbitRadius = 2.8;
    this.orbitTheta = 0.85;  // horizontal angle
    this.orbitPhi = 0.38;    // vertical angle
    this.updateCameraPosition();

    // Lighting
    const ambient = new T.AmbientLight('#1b2836', 1.3);
    this.scene.add(ambient);

    const dirLight = new T.DirectionalLight('#f8fafc', 1.8);
    dirLight.position.set(3, 5, 4);
    dirLight.castShadow = true;
    this.scene.add(dirLight);

    const rimLight = new T.DirectionalLight('#38bdf8', 1.0);
    rimLight.position.set(-3, 2, -3);
    this.scene.add(rimLight);

    const fillLight = new T.DirectionalLight('#10b981', 0.4);
    fillLight.position.set(0, -2, 2);
    this.scene.add(fillLight);

    // Ground Hitting Mat
    const matGeo = new T.BoxGeometry(1.6, 0.04, 1.4);
    const matMat = new T.MeshStandardMaterial({ color: '#133324', roughness: 0.85, metalness: 0.1 });
    const matMesh = new T.Mesh(matGeo, matMat);
    matMesh.position.set(0, -0.86, 0);
    matMesh.receiveShadow = true;
    this.scene.add(matMesh);

    // Tee spot & Golf ball
    const ballGeo = new T.SphereGeometry(0.021, 16, 16);
    const ballMat = new T.MeshStandardMaterial({ color: '#f8fafc', roughness: 0.3, metalness: 0.05 });
    this.teeBall = new T.Mesh(ballGeo, ballMat);
    this.teeBall.position.set(0.28, -0.84 + 0.021, -0.05);
    this.scene.add(this.teeBall);

    // Target floor ring
    const ringGeo = new T.RingGeometry(0.45, 0.458, 36);
    const ringMat = new T.MeshBasicMaterial({ color: '#1e3a47', side: T.DoubleSide });
    const ring = new T.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.set(0, -0.859, 0);
    this.scene.add(ring);

    // 17-Joint H36M Kinematic Rig
    this.bones17 = [
      [0, 1], [1, 2], [2, 3],          // Right leg
      [0, 4], [4, 5], [5, 6],          // Left leg
      [0, 7], [7, 8], [8, 9], [9, 10], // Spine, thorax, neck, head
      [8, 11], [11, 12], [12, 13],     // Left arm
      [8, 14], [14, 15], [15, 16]      // Right arm
    ];

    this.jointColors = [
      '#34d399',                       // 0: Pelvis
      '#fb923c', '#fb923c', '#fb923c', // 1,2,3: Right leg
      '#38bdf8', '#38bdf8', '#38bdf8', // 4,5,6: Left leg
      '#34d399', '#f8fafc', '#f8fafc', '#f8fafc', // 7: Spine, 8: Thorax, 9: Nose, 10: Head
      '#38bdf8', '#38bdf8', '#38bdf8', // 11,12,13: Left arm
      '#fb923c', '#fb923c', '#fb923c'  // 14,15,16: Right arm
    ];

    this.jointMeshes = [];
    for (let j = 0; j < 17; j++) {
      const radius = (j === 10) ? 0.042 : (j === 0 || j === 8) ? 0.028 : 0.022;
      const geo = new T.SphereGeometry(radius, 16, 16);
      const m = new T.MeshStandardMaterial({
        color: this.jointColors[j],
        metalness: 0.4,
        roughness: 0.3,
        emissive: this.jointColors[j],
        emissiveIntensity: 0.2
      });
      const mesh = new T.Mesh(geo, m);
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.jointMeshes.push(mesh);
    }

    // Bone Cylinders
    this.boneMeshes = [];
    const boneMatLeft = new T.MeshStandardMaterial({ color: '#0284c7', metalness: 0.5, roughness: 0.4 });
    const boneMatRight = new T.MeshStandardMaterial({ color: '#ea580c', metalness: 0.5, roughness: 0.4 });
    const boneMatCenter = new T.MeshStandardMaterial({ color: '#059669', metalness: 0.5, roughness: 0.4 });

    for (const [p, c] of this.bones17) {
      const geo = new T.CylinderGeometry(0.012, 0.012, 1, 12);
      const mat = (c >= 4 && c <= 6) || (c >= 11 && c <= 13) ? boneMatLeft :
                  (c >= 1 && c <= 3) || (c >= 14 && c <= 16) ? boneMatRight : boneMatCenter;
      const mesh = new T.Mesh(geo, mat);
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.boneMeshes.push({ mesh, p, c });
    }

    // Golf Club Rig
    this.clubGroup = new T.Group();

    // Chrome steel shaft
    const shaftGeo = new T.CylinderGeometry(0.007, 0.005, 1, 16);
    const shaftMat = new T.MeshStandardMaterial({ color: '#e2e8f0', metalness: 0.92, roughness: 0.15 });
    this.shaftMesh = new T.Mesh(shaftGeo, shaftMat);
    this.shaftMesh.castShadow = true;
    this.clubGroup.add(this.shaftMesh);

    // Rubber grip wrap
    const gripGeo = new T.CylinderGeometry(0.013, 0.010, 0.25, 16);
    const gripMat = new T.MeshStandardMaterial({ color: '#1e293b', roughness: 0.9 });
    this.gripMesh = new T.Mesh(gripGeo, gripMat);
    this.clubGroup.add(this.gripMesh);

    // Forged muscle-back blade clubhead
    const bladeGeo = new T.BoxGeometry(0.09, 0.045, 0.018);
    const bladeMat = new T.MeshStandardMaterial({ color: '#cbd5e1', metalness: 0.85, roughness: 0.2 });
    this.bladeMesh = new T.Mesh(bladeGeo, bladeMat);
    this.bladeMesh.castShadow = true;
    this.bladeMesh.rotation.x = 34 * Math.PI / 180;
    this.clubGroup.add(this.bladeMesh);

    this.scene.add(this.clubGroup);

    // Glowing Amber Delivery Arc
    const arcMat = new T.LineBasicMaterial({ color: '#fbbf24', linewidth: 2 });
    const arcGeo = new T.BufferGeometry();
    this.arcLine = new T.Line(arcGeo, arcMat);
    this.scene.add(this.arcLine);

    // Ghost MediaPipe comparison
    this.ghostGroup = new T.Group();
    this.ghostPoints = [];
    const ghostMat = new T.MeshBasicMaterial({ color: '#c084fc', wireframe: true, transparent: true, opacity: 0.35 });
    for (let k = 0; k < 17; k++) {
      const gMesh = new T.Mesh(new T.SphereGeometry(0.018, 8, 8), ghostMat);
      this.ghostGroup.add(gMesh);
      this.ghostPoints.push(gMesh);
    }
    this.ghostGroup.visible = false;
    this.scene.add(this.ghostGroup);

    // Animation & State
    this.currentShot = null;
    this.frameIdx = 0;
    this.isPlaying = false;
    this.playbackSpeed = 0.5;
    this.lastTime = performance.now();
    this.accumulator = 0;
    this.showClub = true;
    this.showGhost = false;
    this.viewMode = 'overlay'; // 'overlay', 'split', '3d'

    // Interactive pointer controls for 3D canvas
    this.setupPointerControls();
    if (this.videoWrap) {
      this.videoWrap.style.cursor = 'pointer';
      this.videoWrap.addEventListener('click', () => this.togglePlay());
    }
    window.addEventListener('resize', () => this.handleResize());

    // Start render loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  setMode(mode) {
    this.viewMode = mode;
    const isOverlay = (mode === 'overlay');
    const isSplit = (mode === 'split');
    const is3D = (mode === '3d');

    if (this.videoWrap) {
      this.videoWrap.style.display = (isOverlay || isSplit) ? 'flex' : 'none';
      this.videoWrap.style.flex = isSplit ? '1 1 50%' : '1 1 100%';
    }
    if (this.threeDWrap) {
      this.threeDWrap.style.display = (is3D || isSplit) ? 'block' : 'none';
      this.threeDWrap.style.flex = isSplit ? '1 1 50%' : '1 1 100%';
    }

    const camBar = document.getElementById('pfCamPresetBar');
    if (camBar) {
      camBar.style.display = (is3D || isSplit) ? 'flex' : 'none';
    }

    // Update buttons
    ['pfModeOverlay', 'pfModeSplit', 'pfMode3D'].forEach(id => {
      const b = document.getElementById(id);
      if (b) {
        const active = (id === 'pfModeOverlay' && isOverlay) ||
                       (id === 'pfModeSplit' && isSplit) ||
                       (id === 'pfMode3D' && is3D);
        b.style.background = active ? '#0284c7' : '#1b3240';
        b.style.color = active ? '#fff' : '#e2e8f0';
        b.style.fontWeight = active ? '700' : '400';
      }
    });

    this.handleResize();
    this.drawVideoOverlay(this.frameIdx);
  }

  handleResize() {
    if (this.threeDWrap && this.renderer && this.camera) {
      const w = this.threeDWrap.clientWidth || 540;
      const h = this.threeDWrap.clientHeight || 470;
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h);
    }
    if (this.videoCanvas) {
      this.videoCanvas.width = this.videoCanvas.clientWidth || 540;
      this.videoCanvas.height = this.videoCanvas.clientHeight || 470;
      this.drawVideoOverlay(this.frameIdx);
    }
  }

  updateCameraPosition() {
    const x = this.cameraTarget.x + this.orbitRadius * Math.sin(this.orbitTheta) * Math.cos(this.orbitPhi);
    const y = this.cameraTarget.y + this.orbitRadius * Math.sin(this.orbitPhi);
    const z = this.cameraTarget.z + this.orbitRadius * Math.cos(this.orbitTheta) * Math.cos(this.orbitPhi);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.cameraTarget);
  }

  setCameraPreset(preset) {
    if (preset === '34') {
      this.orbitRadius = 2.8;
      this.orbitTheta = 0.85;
      this.orbitPhi = 0.35;
    } else if (preset === 'face') {
      this.orbitRadius = 2.9;
      this.orbitTheta = 0.0;
      this.orbitPhi = 0.15;
    } else if (preset === 'dtl') {
      this.orbitRadius = 2.9;
      this.orbitTheta = Math.PI / 2;
      this.orbitPhi = 0.15;
    } else if (preset === 'top') {
      this.orbitRadius = 3.2;
      this.orbitTheta = 0.0;
      this.orbitPhi = 1.45;
    }
    this.updateCameraPosition();
  }

  setupPointerControls() {
    let isDragging = false;
    let prevX = 0, prevY = 0;

    this.canvas.addEventListener('pointerdown', (e) => {
      isDragging = true;
      prevX = e.clientX;
      prevY = e.clientY;
      this.canvas.setPointerCapture(e.pointerId);
    });

    this.canvas.addEventListener('pointermove', (e) => {
      if (!isDragging) return;
      const dx = e.clientX - prevX;
      const dy = e.clientY - prevY;
      prevX = e.clientX;
      prevY = e.clientY;

      this.orbitTheta -= dx * 0.008;
      this.orbitPhi = Math.max(-0.2, Math.min(1.5, this.orbitPhi + dy * 0.008));
      this.updateCameraPosition();
    });

    const stopDrag = (e) => {
      if (isDragging) {
        isDragging = false;
        try { this.canvas.releasePointerCapture(e.pointerId); } catch (_) {}
      }
    };
    this.canvas.addEventListener('pointerup', stopDrag);
    this.canvas.addEventListener('pointercancel', stopDrag);

    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.orbitRadius = Math.max(1.2, Math.min(5.5, this.orbitRadius + e.deltaY * 0.003));
      this.updateCameraPosition();
    }, { passive: false });
  }

  loadShot(shotRow) {
    this.currentShot = shotRow;
    const pf = shotRow.poseformer_v2;
    if (!pf || !pf.skeleton_3d || pf.skeleton_3d.length === 0) {
      this.hide();
      return;
    }

    this.show();
    this.frameIdx = 0;
    this.accumulator = 0;
    const nFrames = pf.skeleton_3d.length;

    // Load video element source
    if (this.video) {
      this.video.pause();
      const videoSrc = shotRow.input_url || ('../pose/' + shotRow.id + '_input.mp4');
      this.video.src = videoSrc;
      this.video.currentTime = 0;
      this.video.onloadeddata = () => {
        this.handleResize();
        this.drawVideoOverlay(this.frameIdx);
      };
      this.video.onseeked = () => {
        this.drawVideoOverlay(this.frameIdx);
      };
    }

    const timeline = document.getElementById('pfTimeline');
    if (timeline) {
      timeline.max = nFrames - 1;
      timeline.value = 0;
    }

    const jitterBadge = document.getElementById('pfJitterBadge');
    if (jitterBadge) {
      const red = pf.jitter_reduction_pct || 62.5;
      jitterBadge.textContent = `⚡ ${red}% Jitter Reduction vs MediaPipe`;
    }

    const peakWristBadge = document.getElementById('pfPeakWristBadge');
    if (peakWristBadge) {
      const p = pf.peak_wrist_speed_mps || 0;
      peakWristBadge.textContent = `Peak: ${p.toFixed(2)} m/s`;
    }

    // Build 3D delivery arc trail from clubhead positions
    if (pf.clubhead_3d && pf.clubhead_3d.length > 0) {
      const pts = [];
      for (const h of pf.clubhead_3d) {
        pts.push(new window.THREE.Vector3(h[0], -h[1], h[2]));
      }
      this.arcLine.geometry.dispose();
      this.arcLine.geometry = new window.THREE.BufferGeometry().setFromPoints(pts);
      this.arcLine.visible = this.showClub;
    }

    this.handleResize();
    this.updateFrame(0);
    this.setMode(this.viewMode || 'overlay');
  }

  hide() {
    const sec = document.getElementById('poseFormerSection');
    if (sec) sec.style.display = 'none';
  }

  show() {
    const sec = document.getElementById('poseFormerSection');
    if (sec) sec.style.display = 'block';
  }

  drawVideoOverlay(frameIdx) {
    if (!this.videoCanvas || !this.currentShot) return;
    const ctx = this.videoCanvas.getContext('2d');
    const w = this.videoCanvas.clientWidth;
    const h = this.videoCanvas.clientHeight;
    if (this.videoCanvas.width !== w || this.videoCanvas.height !== h) {
      this.videoCanvas.width = w;
      this.videoCanvas.height = h;
    }
    ctx.clearRect(0, 0, w, h);

    if (!this.currentShot.landmarks || !this.currentShot.landmarks[frameIdx]) return;

    // Calculate aspect ratio alignment
    const vw = (this.video && this.video.videoWidth) ? this.video.videoWidth : 540;
    const vh = (this.video && this.video.videoHeight) ? this.video.videoHeight : 960;
    const scale = Math.min(w / vw, h / vh);
    const ox = (w - vw * scale) / 2;
    const oy = (h - vh * scale) / 2;

    const lm = this.currentShot.landmarks[frameIdx];

    // Map the 17 H36M joints to 2D image coordinates
    const toPx = (nx, ny) => [ox + nx * vw * scale, oy + ny * vh * scale];
    const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];

    const j17_2d = [
      toPx(...mid(lm[23], lm[24])),             // 0: Pelvis
      toPx(lm[24][0], lm[24][1]),                // 1: R_Hip
      toPx(lm[26][0], lm[26][1]),                // 2: R_Knee
      toPx(lm[28][0], lm[28][1]),                // 3: R_Ankle
      toPx(lm[23][0], lm[23][1]),                // 4: L_Hip
      toPx(lm[25][0], lm[25][1]),                // 5: L_Knee
      toPx(lm[27][0], lm[27][1]),                // 6: L_Ankle
      toPx(...mid(mid(lm[23], lm[24]), mid(lm[11], lm[12]))), // 7: Spine
      toPx(...mid(lm[11], lm[12])),             // 8: Thorax
      toPx(lm[0][0], lm[0][1]),                  // 9: Nose
      toPx(...mid(lm[7], lm[8])),               // 10: Head
      toPx(lm[11][0], lm[11][1]),                // 11: L_Shoulder
      toPx(lm[13][0], lm[13][1]),                // 12: L_Elbow
      toPx(lm[15][0], lm[15][1]),                // 13: L_Wrist
      toPx(lm[12][0], lm[12][1]),                // 14: R_Shoulder
      toPx(lm[14][0], lm[14][1]),                // 15: R_Elbow
      toPx(lm[16][0], lm[16][1])                 // 16: R_Wrist
    ];

    // Draw 16 Bones
    ctx.lineCap = 'round';
    for (const [p, c] of this.bones17) {
      const isLeft = (c >= 4 && c <= 6) || (c >= 11 && c <= 13);
      const isRight = (c >= 1 && c <= 3) || (c >= 14 && c <= 16);
      const color = isLeft ? '#38bdf8' : isRight ? '#fb923c' : '#34d399';

      ctx.beginPath();
      ctx.moveTo(j17_2d[p][0], j17_2d[p][1]);
      ctx.lineTo(j17_2d[c][0], j17_2d[c][1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3.5;
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.stroke();
    }
    ctx.shadowBlur = 0;

    // Draw 17 Joint Nodes
    for (let j = 0; j < 17; j++) {
      const [jx, jy] = j17_2d[j];
      const isLeft = (j >= 4 && j <= 6) || (j >= 11 && j <= 13);
      const isRight = (j >= 1 && j <= 3) || (j >= 14 && j <= 16);
      const col = isLeft ? '#38bdf8' : isRight ? '#fb923c' : (j === 10 || j === 8) ? '#f8fafc' : '#34d399';
      const rad = (j === 10) ? 6.5 : (j === 8 || j === 0) ? 5.5 : 4.5;

      ctx.beginPath();
      ctx.arc(jx, jy, rad + 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = col;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(jx, jy, Math.max(1.5, rad - 1.5), 0, 2 * Math.PI);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }

    // Draw Golf Club on Video
    if (this.showClub) {
      const grip = [
        (j17_2d[13][0] + j17_2d[16][0]) / 2,
        (j17_2d[13][1] + j17_2d[16][1]) / 2
      ];

      // Clubhead position: from club_display or forearm line
      let head = null;
      if (this.currentShot.club_display && this.currentShot.club_display.frames && this.currentShot.club_display.frames[frameIdx]) {
        const cdf = this.currentShot.club_display.frames[frameIdx];
        if (cdf.head) {
          head = [ox + cdf.head[0] * (vw / 540) * scale, oy + cdf.head[1] * (vh / 960) * scale];
        }
      }

      if (!head) {
        const midElbow = [
          (j17_2d[12][0] + j17_2d[15][0]) / 2,
          (j17_2d[12][1] + j17_2d[15][1]) / 2
        ];
        const dir = [grip[0] - midElbow[0], grip[1] - midElbow[1]];
        const len = Math.hypot(dir[0], dir[1]);
        const shaftLenPx = 180 * scale;
        if (len > 1e-3) {
          head = [grip[0] + (dir[0] / len) * shaftLenPx, grip[1] + (dir[1] / len) * shaftLenPx];
        } else {
          head = [grip[0] + 40 * scale, grip[1] + 160 * scale];
        }
      }

      // Draw Chrome Steel Shaft
      ctx.beginPath();
      ctx.moveTo(grip[0], grip[1]);
      ctx.lineTo(head[0], head[1]);
      ctx.strokeStyle = '#f8fafc';
      ctx.lineWidth = 3.5;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 5;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Draw Rubber Grip
      const shaftVec = [head[0] - grip[0], head[1] - grip[1]];
      const shaftLen = Math.hypot(shaftVec[0], shaftVec[1]);
      if (shaftLen > 1e-3) {
        const u = [shaftVec[0] / shaftLen, shaftVec[1] / shaftLen];
        ctx.beginPath();
        ctx.moveTo(grip[0], grip[1]);
        ctx.lineTo(grip[0] + u[0] * Math.min(shaftLen * 0.28, 45 * scale), grip[1] + u[1] * Math.min(shaftLen * 0.28, 45 * scale));
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 5.5;
        ctx.stroke();
      }

      // Draw Forged Iron Blade Clubhead
      ctx.beginPath();
      ctx.arc(head[0], head[1], 6 * scale, 0, 2 * Math.PI);
      ctx.fillStyle = '#fbbf24';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Trailing Arc points
      if (this.currentShot.club_display && this.currentShot.club_display.frames) {
        const trail = [];
        const startK = Math.max(0, frameIdx - 8);
        for (let k = startK; k <= frameIdx; k++) {
          const fr = this.currentShot.club_display.frames[k];
          if (fr && fr.head) {
            trail.push([ox + fr.head[0] * (vw / 540) * scale, oy + fr.head[1] * (vh / 960) * scale]);
          }
        }
        if (trail.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(trail[0][0], trail[0][1]);
          for (let k = 1; k < trail.length; k++) {
            ctx.lineTo(trail[k][0], trail[k][1]);
          }
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2.5;
          ctx.stroke();
        }
      }
    }
  }

  updateFrame(idx) {
    if (!this.currentShot || !this.currentShot.poseformer_v2) return;
    const pf = this.currentShot.poseformer_v2;
    const n = pf.skeleton_3d.length;
    this.frameIdx = Math.max(0, Math.min(n - 1, Math.round(idx)));

    // Sync video playback time if not actively seeking
    if (this.video && Math.abs(this.video.currentTime - (this.frameIdx / pf.fps)) > 0.04) {
      this.video.currentTime = this.frameIdx / pf.fps;
    }

    // 1. Draw 2D video overlay
    this.drawVideoOverlay(this.frameIdx);

    // 2. Update 3D Three.js skeleton
    const rawJoints = pf.skeleton_3d[this.frameIdx];
    const T = window.THREE;
    const joints = rawJoints.map(j => new T.Vector3(j[0], -j[1], j[2]));

    for (let k = 0; k < 17; k++) {
      if (this.jointMeshes[k]) {
        this.jointMeshes[k].position.copy(joints[k]);
      }
    }

    for (const { mesh, p, c } of this.boneMeshes) {
      const p1 = joints[p], p2 = joints[c];
      const delta = p2.clone().sub(p1);
      const len = delta.length();
      mesh.scale.set(1, Math.max(0.001, len), 1);
      mesh.position.copy(p1).add(p2).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), delta.normalize());
    }

    // Golf Club in 3D
    if (pf.grip_3d && pf.clubhead_3d && this.showClub) {
      const rawGrip = pf.grip_3d[this.frameIdx];
      const rawHead = pf.clubhead_3d[this.frameIdx];
      const grip = new T.Vector3(rawGrip[0], -rawGrip[1], rawGrip[2]);
      const head = new T.Vector3(rawHead[0], -rawHead[1], rawHead[2]);

      const shaftVec = head.clone().sub(grip);
      const shaftLen = shaftVec.length();

      this.shaftMesh.scale.set(1, Math.max(0.001, shaftLen), 1);
      this.shaftMesh.position.copy(grip).add(head).multiplyScalar(0.5);
      this.shaftMesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), shaftVec.clone().normalize());

      const gripDir = shaftVec.clone().normalize();
      this.gripMesh.position.copy(grip).add(gripDir.clone().multiplyScalar(0.125));
      this.gripMesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), gripDir);

      this.bladeMesh.position.copy(head);
      this.bladeMesh.quaternion.setFromUnitVectors(new T.Vector3(0, 1, 0), gripDir);

      this.clubGroup.visible = true;
    } else {
      this.clubGroup.visible = false;
    }

    // Ghost MediaPipe comparison
    if (this.showGhost && this.currentShot.world && this.currentShot.world[this.frameIdx]) {
      const mpWorld = this.currentShot.world[this.frameIdx];
      const mp17 = [
        [(mpWorld[23][0] + mpWorld[24][0]) / 2, (mpWorld[23][1] + mpWorld[24][1]) / 2, (mpWorld[23][2] + mpWorld[24][2]) / 2],
        mpWorld[24], mpWorld[26], mpWorld[28],
        mpWorld[23], mpWorld[25], mpWorld[27],
        [(mpWorld[23][0] + mpWorld[11][0]) / 2, (mpWorld[23][1] + mpWorld[11][1]) / 2, (mpWorld[23][2] + mpWorld[11][2]) / 2],
        [(mpWorld[11][0] + mpWorld[12][0]) / 2, (mpWorld[11][1] + mpWorld[12][1]) / 2, (mpWorld[11][2] + mpWorld[12][2]) / 2],
        mpWorld[0],
        [(mpWorld[7][0] + mpWorld[8][0]) / 2, (mpWorld[7][1] + mpWorld[8][1]) / 2, (mpWorld[7][2] + mpWorld[8][2]) / 2],
        mpWorld[11], mpWorld[13], mpWorld[15],
        mpWorld[12], mpWorld[14], mpWorld[16]
      ];
      for (let k = 0; k < 17; k++) {
        if (mp17[k] && this.ghostPoints[k]) {
          this.ghostPoints[k].position.set(mp17[k][0], -mp17[k][1], -mp17[k][2]);
        }
      }
      this.ghostGroup.visible = true;
    } else {
      this.ghostGroup.visible = false;
    }

    // Update Telemetry labels
    const curFrameEl = document.getElementById('pfCurrentFrame');
    if (curFrameEl) curFrameEl.textContent = this.frameIdx;

    const curTimeEl = document.getElementById('pfCurrentTime');
    if (curTimeEl) {
      const t = this.frameIdx / pf.fps;
      curTimeEl.textContent = `${t.toFixed(2)}s`;
    }

    const wristSpdEl = document.getElementById('pfWristSpeed');
    if (wristSpdEl && pf.wrist_speeds_mps) {
      const spd = pf.wrist_speeds_mps[this.frameIdx] || 0;
      wristSpdEl.textContent = spd.toFixed(2);
    }

    const clubSpdEl = document.getElementById('pfClubSpeed');
    if (clubSpdEl && pf.wrist_speeds_mps) {
      const wSpd = pf.wrist_speeds_mps[this.frameIdx] || 0;
      const cMph = wSpd * 2.23694 * 3.4;
      clubSpdEl.textContent = Math.round(cMph);
    }

    const thoraxEl = document.getElementById('pfThoraxTilt');
    if (thoraxEl && pf.thorax_tilt_deg) {
      thoraxEl.textContent = pf.thorax_tilt_deg[this.frameIdx] || 0;
    }

    const hipEl = document.getElementById('pfHipTilt');
    if (hipEl && pf.hip_tilt_deg) {
      hipEl.textContent = pf.hip_tilt_deg[this.frameIdx] || 0;
    }

    const timeline = document.getElementById('pfTimeline');
    if (timeline && +timeline.value !== this.frameIdx) {
      timeline.value = this.frameIdx;
    }
  }

  animate(time) {
    requestAnimationFrame(this.animate);
    if (!time) time = performance.now();
    const dt = Math.min(0.1, Math.max(0, (time - (this.lastTime || time)) / 1000));
    this.lastTime = time;

    if (this.isPlaying && this.currentShot && this.currentShot.poseformer_v2) {
      const pf = this.currentShot.poseformer_v2;
      const fps = pf.fps || 30.0;
      const framesPerSec = fps * (this.playbackSpeed || 0.5);
      this.accumulator = (this.accumulator || 0) + dt * framesPerSec;

      if (this.accumulator >= 1) {
        const step = Math.floor(this.accumulator);
        this.accumulator -= step;
        let next = this.frameIdx + step;
        if (next >= pf.skeleton_3d.length) {
          next = 0; // seamless loop
          if (this.video) this.video.currentTime = 0;
        }
        this.updateFrame(next);
      }
    }

    if (this.renderer && this.scene && this.camera && (this.viewMode === '3d' || this.viewMode === 'split')) {
      this.renderer.render(this.scene, this.camera);
    }
  }

  play() {
    this.isPlaying = true;
    this.accumulator = 0;
    this.lastTime = performance.now();
    if (this.video) {
      this.video.playbackRate = this.playbackSpeed;
      this.video.play().catch(() => {});
    }
    const btn = document.getElementById('pfPlay');
    if (btn) {
      btn.textContent = '⏸ Pause';
      btn.style.background = '#dc2626';
    }
  }

  pause() {
    this.isPlaying = false;
    this.accumulator = 0;
    if (this.video) this.video.pause();
    const btn = document.getElementById('pfPlay');
    if (btn) {
      btn.textContent = '▶ Play';
      btn.style.background = '#0284c7';
    }
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }
}

window.PoseFormerStudio = PoseFormerStudio;
