/* Local Three.js display rig. Anatomical human golfer, forged iron club, and physics-gated 3D ball flight.
   Dual trajectory comparison: TrackMan Ground Truth (cyan) vs Video Launch Prediction (purple).
   Never returned as scientific measurements or fed to prediction models. */
window.SwingRange = class SwingRange {
  constructor(canvas) {
    this.canvas=canvas;this.ready=false;this.lastShot=null;this.view='trackman';this.dragYaw=0;this.dragPitch=0;
    if(!window.THREE) return;
    const T=THREE;this.T=T;
    try {this.renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false});} catch(e){this.error=e.message;return;}
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));this.renderer.shadowMap.enabled=true;this.renderer.shadowMap.type=T.PCFSoftShadowMap;this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=.94;
    this.scene=new T.Scene();this.scene.background=new T.Color('#b8d7e8');this.scene.fog=new T.Fog('#b8d7e8',65,220);
    this.camera=new T.PerspectiveCamera(48,540/470,.05,320);
    this.scene.add(new T.HemisphereLight('#f2f7ff','#3b5b2e',2.2));
    const sun=new T.DirectionalLight('#fff2d6',3.4);sun.position.set(-8,16,6);sun.castShadow=true;sun.shadow.mapSize.set(1024,1024);sun.shadow.camera.left=-8;sun.shadow.camera.right=8;sun.shadow.camera.top=8;sun.shadow.camera.bottom=-8;sun.shadow.bias=-.0004;this.scene.add(sun);
    this.materials={
      shirt:this.mat('#22675d',18),
      pants:this.mat('#505b62',12),
      skin:this.mat('#bd8969',14),
      shoe:this.mat('#f8f9fa',32),
      shoeTrim:this.mat('#1e2428',20),
      cap:this.mat('#1a2b35',18),
      shaft:new T.MeshStandardMaterial({color:'#e2e8f0',metalness:.95,roughness:.12}),
      grip:new T.MeshStandardMaterial({color:'#171717',metalness:.1,roughness:.88}),
      ferrule:new T.MeshStandardMaterial({color:'#111111',metalness:.2,roughness:.3}),
      ironFace:new T.MeshStandardMaterial({color:'#cbd5e1',metalness:.88,roughness:.22}),
      ironBody:new T.MeshStandardMaterial({color:'#94a3b8',metalness:.94,roughness:.14}),
      ballTruth:new T.MeshStandardMaterial({color:'#38bdf8',metalness:.2,roughness:.3,emissive:'#0284c7',emissiveIntensity:.35}),
      ballPred:new T.MeshStandardMaterial({color:'#c084fc',metalness:.2,roughness:.3,emissive:'#7c3aed',emissiveIntensity:.35}),
      turfMat:new T.MeshStandardMaterial({color:'#214e34',roughness:.8,metalness:.05}),
      hittingStrip:new T.MeshStandardMaterial({color:'#286343',roughness:.75,metalness:.05}),
      sand:new T.MeshStandardMaterial({color:'#dfce9e',roughness:.92,metalness:.02}),
      glove:this.mat('#faf8f5',22)
    };
    this.ballRadius=.028;
    this.hittingStripTop=.049;
    this.ballOrigin=new T.Vector3(.48,this.hittingStripTop+this.ballRadius,-.05);
    this.buildRange();this.buildGolfer();
    
    // TrackMan Truth Ball & Trace (Cyan #38bdf8)
    this.ballTruth=this.mesh(new T.SphereGeometry(this.ballRadius,16,14),this.materials.ballTruth);this.ballTruth.castShadow=true;this.scene.add(this.ballTruth);
    this.trailSteps={air:120,hop:24,roll:48};
    const trailSegments=this.trailSteps.air+this.trailSteps.hop+this.trailSteps.roll;
    this.traceGeometryTruth=new T.BufferGeometry();this.traceGeometryTruth.setAttribute('position',new T.Float32BufferAttribute(new Float32Array((trailSegments+1)*3),3));
    this.traceTruth=new T.Line(this.traceGeometryTruth,new T.LineBasicMaterial({color:'#38bdf8',transparent:true,opacity:.95}));
    this.traceTruth.frustumCulled=false;this.scene.add(this.traceTruth);

    // Best Model Predicted Ball & Trace (Purple #c084fc)
    this.ballPred=this.mesh(new T.SphereGeometry(this.ballRadius,16,14),this.materials.ballPred);this.ballPred.castShadow=true;this.scene.add(this.ballPred);
    this.traceGeometryPred=new T.BufferGeometry();this.traceGeometryPred.setAttribute('position',new T.Float32BufferAttribute(new Float32Array((trailSegments+1)*3),3));
    this.tracePred=new T.Line(this.traceGeometryPred,new T.LineBasicMaterial({color:'#c084fc',transparent:true,opacity:.95}));
    this.tracePred.frustumCulled=false;this.scene.add(this.tracePred);

    // Thick 3D glowing tracer ribbons (Toptracer broadcast style in WebGL)
    const makeRibbonGeom=()=>{
      const g=new T.BufferGeometry();
      g.setAttribute('position',new T.BufferAttribute(new Float32Array((trailSegments+1)*2*3),3));
      const indices=new Uint16Array(trailSegments*6);let idx=0;
      for(let j=0;j<trailSegments;j++){
        const a=j*2,b=a+1,c=a+2,d=a+3;
        indices[idx++]=a;indices[idx++]=b;indices[idx++]=c;
        indices[idx++]=b;indices[idx++]=d;indices[idx++]=c;
      }
      g.setIndex(new T.BufferAttribute(indices,1));g.setDrawRange(0,0);
      return g;
    };
    this.ribbonGeomTruth=makeRibbonGeom();
    this.ribbonTruth=new T.Mesh(this.ribbonGeomTruth,new T.MeshBasicMaterial({color:'#38bdf8',side:T.DoubleSide,transparent:true,opacity:.98}));
    this.ribbonTruth.frustumCulled=false;this.scene.add(this.ribbonTruth);

    this.ribbonGeomPred=makeRibbonGeom();
    this.ribbonPred=new T.Mesh(this.ribbonGeomPred,new T.MeshBasicMaterial({color:'#c084fc',side:T.DoubleSide,transparent:true,opacity:.98}));
    this.ribbonPred.frustumCulled=false;this.scene.add(this.ribbonPred);

    // Landing spot bullseye rings on the turf for direct visual landing comparison
    this.landRingTruth=this.mesh(new T.RingGeometry(.3,1.3,32),new T.MeshBasicMaterial({color:'#38bdf8',side:T.DoubleSide,transparent:true,opacity:.95}));
    this.landRingTruth.rotation.x=-Math.PI/2;this.landRingTruth.visible=false;this.scene.add(this.landRingTruth);
    this.discTruth=this.mesh(new T.CircleGeometry(.32,24),new T.MeshBasicMaterial({color:'#0284c7',side:T.DoubleSide,transparent:true,opacity:.92}));
    this.discTruth.rotation.x=-Math.PI/2;this.discTruth.visible=false;this.scene.add(this.discTruth);
    this.outerRingTruth=this.mesh(new T.RingGeometry(1.5,2.3,32),new T.MeshBasicMaterial({color:'#38bdf8',side:T.DoubleSide,transparent:true,opacity:.45}));
    this.outerRingTruth.rotation.x=-Math.PI/2;this.outerRingTruth.visible=false;this.scene.add(this.outerRingTruth);

    this.landRingPred=this.mesh(new T.RingGeometry(.3,1.3,32),new T.MeshBasicMaterial({color:'#c084fc',side:T.DoubleSide,transparent:true,opacity:.95}));
    this.landRingPred.rotation.x=-Math.PI/2;this.landRingPred.visible=false;this.scene.add(this.landRingPred);
    this.discPred=this.mesh(new T.CircleGeometry(.32,24),new T.MeshBasicMaterial({color:'#7c3aed',side:T.DoubleSide,transparent:true,opacity:.92}));
    this.discPred.rotation.x=-Math.PI/2;this.discPred.visible=false;this.scene.add(this.discPred);
    this.outerRingPred=this.mesh(new T.RingGeometry(1.5,2.3,32),new T.MeshBasicMaterial({color:'#c084fc',side:T.DoubleSide,transparent:true,opacity:.45}));
    this.outerRingPred.rotation.x=-Math.PI/2;this.outerRingPred.visible=false;this.scene.add(this.outerRingPred);

    // Final total distance resting markers (where ball rolls to a stop)
    this.totalRingTruth=this.mesh(new T.RingGeometry(.18,.55,24),new T.MeshBasicMaterial({color:'#38bdf8',side:T.DoubleSide,transparent:true,opacity:.85}));
    this.totalRingTruth.rotation.x=-Math.PI/2;this.totalRingTruth.visible=false;this.scene.add(this.totalRingTruth);
    this.totalRingPred=this.mesh(new T.RingGeometry(.18,.55,24),new T.MeshBasicMaterial({color:'#c084fc',side:T.DoubleSide,transparent:true,opacity:.85}));
    this.totalRingPred.rotation.x=-Math.PI/2;this.totalRingPred.visible=false;this.scene.add(this.totalRingPred);

    this.cameraMode='side_wide';

    // Legacy aliases for existing references
    this.ball=this.ballPred;
    this.trace=this.tracePred;
    this.traceGeometry=this.traceGeometryPred;

    if(typeof document!=='undefined'){
      this.markerOverlay=document.getElementById('rangeMarkers');
    }

    this.zoomOffset={x:0,y:0};
    this.bindCameraControls(canvas);
    this.ready=true;
  }
  updateCameraProjection(){
    const {x,y}=this.zoomOffset;
    const width=this.camera.aspect;
    if(x||y)this.camera.setViewOffset(width,1,-x*width*.5,y*.5,width,1);
    else this.camera.clearViewOffset();
  }
  zoomCamera(factor,clientX,clientY){
    const previous=this.camera.zoom;
    const zoom=THREE.MathUtils.clamp(previous*factor,.45,4);
    if(zoom===previous)return;
    const bounds=Number.isFinite(clientX)&&Number.isFinite(clientY)
      ?this.canvas.getBoundingClientRect():null;
    const x=bounds?.width?2*(clientX-bounds.left)/bounds.width-1:0;
    const y=bounds?.height?1-2*(clientY-bounds.top)/bounds.height:0;
    const scale=zoom/previous;
    // Compensate for the new scale so the point under the pointer stays put.
    this.zoomOffset.x=x+scale*(this.zoomOffset.x-x);
    this.zoomOffset.y=y+scale*(this.zoomOffset.y-y);
    this.camera.zoom=zoom;
    this.updateCameraProjection();
  }
  bindCameraControls(canvas){
    const pointers=new Map();
    const separation=()=>{
      const [a,b]=pointers.values();
      return Math.hypot(b.x-a.x,b.y-a.y);
    };
    canvas.addEventListener('pointerdown',event=>{
      if(event.pointerType==='mouse' && event.button!==0)return;
      pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener('pointermove',event=>{
      const previous=pointers.get(event.pointerId);
      if(!previous)return;
      const before=pointers.size===2?separation():0;
      pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
      if(pointers.size===2){
        if(before>0){
          const [a,b]=pointers.values();
          this.zoomCamera(separation()/before,(a.x+b.x)/2,(a.y+b.y)/2);
        }
      }else if(pointers.size===1){
        this.dragYaw+=(event.clientX-previous.x)*.008;
        const overview=this.cameraMode==='side_wide';
        this.dragPitch=THREE.MathUtils.clamp(this.dragPitch+(event.clientY-previous.y)*.004,
          overview?-1.35:-.35,overview ? .96 : .8);
      }
    });
    for(const type of ['pointerup','pointercancel','lostpointercapture']){
      canvas.addEventListener(type,event=>pointers.delete(event.pointerId));
    }
    let gestureZoom=null;
    canvas.addEventListener('gesturestart',event=>{event.preventDefault();gestureZoom=this.camera.zoom;});
    canvas.addEventListener('gesturechange',event=>{
      event.preventDefault();
      if(gestureZoom!==null)this.zoomCamera(gestureZoom*event.scale/this.camera.zoom,event.clientX,event.clientY);
    });
    canvas.addEventListener('gestureend',event=>{event.preventDefault();gestureZoom=null;});
    canvas.addEventListener('wheel',event=>{
      if(!event.deltaY)return;
      event.preventDefault();
      if(gestureZoom!==null)return;
      const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?canvas.clientHeight:1);
      this.zoomCamera(Math.exp(THREE.MathUtils.clamp(-delta*(event.ctrlKey ? .012 : .0025),-1,1)),event.clientX,event.clientY);
    },{passive:false});
    canvas.style.touchAction='none';
  }
  mat(color,shininess=15){return new THREE.MeshPhongMaterial({color,shininess,flatShading:false})}
  mesh(geometry,material){const m=new THREE.Mesh(geometry,material);m.castShadow=true;m.receiveShadow=true;return m}
  box(size,pos,color){const m=this.mesh(new THREE.BoxGeometry(...size),typeof color==='string'?this.mat(color):color);m.position.set(...pos);this.scene.add(m);return m}
  buildRange(){
    const T=THREE;
    let seed=13471;
    const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
    const hasCanvas=typeof document!=='undefined';
    const surface=(color,texture)=>new T.MeshStandardMaterial({color,map:texture||null,roughness:.94});
    const texture=(canvas)=>{
      const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;
      map.anisotropy=Math.min(8,this.renderer.capabilities.getMaxAnisotropy());return map;
    };
    if(hasCanvas){
      const sky=document.createElement('canvas');sky.width=1024;sky.height=512;
      const ctx=sky.getContext('2d'),gradient=ctx.createLinearGradient(0,0,0,512);
      for(const [u,c] of [[0,'#4384b2'],[.35,'#79adca'],[.50,'#e8dbc0'],[.54,'#b9c5ad'],[1,'#96aa95']])gradient.addColorStop(u,c);
      ctx.fillStyle=gradient;ctx.fillRect(0,0,1024,512);
      const skyMap=texture(sky);skyMap.mapping=T.EquirectangularReflectionMapping;this.scene.background=skyMap;
    }
    this.scene.fog.color.set('#c8d4c3');
    this.box([1800,.1,1800],[0,-.26,-110],'#52653d');
    const {x:teeX,z:teeZ}=this.ballOrigin;
    this.rangePins=[[50,-14],[85,19],[115,-23],[150,9],[185,32],[215,-18],[250,14]].map(([yd,side])=>({
      x:teeX+side,z:teeZ-Math.sqrt((yd*.9144)**2-side**2),yd
    }));
    // Paint a continuous landscape at metre coordinates so greens, their collars,
    // bunkers and the physical flagsticks share exactly the same placement.
    let fieldMap;
    if(hasCanvas){
      const field=document.createElement('canvas');field.width=1536;field.height=2304;
      const ctx=field.getContext('2d');ctx.scale(8,8);ctx.translate(96,258);
      ctx.fillStyle='#48653f';ctx.fillRect(-96,-258,192,288);
      const fairway=new Path2D();
      for(let z=12;z>=-250;z-=2){
        const x=-31+4*Math.sin(z/21)+6*Math.sin(z/55);if(z===12)fairway.moveTo(x,z);else fairway.lineTo(x,z);
      }
      for(let z=-250;z<=12;z+=2)fairway.lineTo(34+4*Math.sin(z/26)+4*Math.cos(z/45),z);
      fairway.closePath();ctx.save();ctx.clip(fairway);ctx.fillStyle='#597e4e';ctx.fillRect(-60,-254,120,270);
      ctx.rotate(.19);
      for(let x=-140;x<130;x+=10){ctx.fillStyle='rgba(174,199,141,.09)';ctx.fillRect(x,-290,5,335);}
      ctx.restore();
      const patch=(x,z,rx,rz,angle,phase)=>{
        const shape=new Path2D();
        for(let i=0;i<=80;i++){
          const a=i*Math.PI/40,r=1+.12*Math.sin(3*a+phase)+.065*Math.cos(2*a-phase);
          const u=rx*r*Math.cos(a),v=rz*r*Math.sin(a);
          const px=x+u*Math.cos(angle)-v*Math.sin(angle),pz=z+u*Math.sin(angle)+v*Math.cos(angle);
          if(i===0)shape.moveTo(px,pz);else shape.lineTo(px,pz);
        }
        shape.closePath();return shape;
      };
      for(const [i,pin] of this.rangePins.entries()){
        const {x,z}=pin,r=4.8+i*.44,angle=.4*Math.sin(i*2);
        for(const [size,color] of [[1.19,'#577d50'],[1.09,'#779c65'],[1,'#86a973']]){
          ctx.fillStyle=color;ctx.fill(patch(x-.8,z+.35,r*1.4*size,r*.88*size,angle,i));
        }
        if([1,3,5,6].includes(i)){
          const sign=i%3===0?-1:1,bx=x+sign*r*1.35,bz=z+r*.82;
          const sand=patch(bx,bz,r*.66,r*.32,angle-sign*.64,i+2);
          ctx.lineWidth=.52;ctx.strokeStyle='#596e40';ctx.stroke(sand);
          const g=ctx.createLinearGradient(bx,bz-r*.37,bx,bz+r*.4);
          g.addColorStop(0,'#ac9d75');g.addColorStop(.35,'#c9b891');g.addColorStop(1,'#ddccaa');
          ctx.fillStyle=g;ctx.fill(sand);
        }
      }
      // Fine variation softens the mowing bands at ground level.
      for(let i=0;i<82000;i++){
        ctx.fillStyle=i%2?'rgba(39,58,29,.12)':'rgba(218,219,157,.13)';
        ctx.fillRect(random()*192-96,random()*288-258,.09+random()*.22,.07);
      }
      fieldMap=texture(field);
    }
    const field=this.mesh(new T.PlaneGeometry(192,288),surface(fieldMap?'#ffffff':'#789358',fieldMap));
    field.rotation.x=-Math.PI/2;field.position.set(0,-.06,-114);this.scene.add(field);
    const dashLength=3.5,dashSpacing=7,start=teeZ-1.3,end=-246;
    const dashCount=1+Math.floor((start-end-dashLength)/dashSpacing);
    const dash=new T.PlaneGeometry(.24,dashLength).rotateX(-Math.PI/2);
    this.centerline=new T.InstancedMesh(dash,new T.MeshBasicMaterial({color:'#ffffff',toneMapped:false}),dashCount);
    const transform=new T.Matrix4();
    for(let i=0;i<dashCount;i++){
      this.centerline.setMatrixAt(i,transform.makeTranslation(teeX,field.position.y+.004,start-i*dashSpacing-dashLength/2));
    }
    this.centerline.visible=false;
    this.scene.add(this.centerline);
    // Distant wooded ridges form a continuous horizon around the practice grounds.
    for(let band=0;band<2;band++){
      const vertices=[],indices=[],n=128;
      for(let i=0;i<=n;i++){
        const a=2*Math.PI*i/n,r=247+65*band+11*Math.sin(a*5+.7);
        const height=12+band*8+5*Math.sin(a*4+band)+4*Math.sin(a*9)+2*Math.cos(a*17);
        for(const [rad,y] of [[r-32,-1],[r,height],[r+24,height*.8]]){
          vertices.push(rad*Math.cos(a),y,-83+rad*Math.sin(a));
        }
        if(i<n)for(let j=0;j<2;j++){
          const k=i*3+j;indices.push(k,k+1,k+3,k+1,k+4,k+3);
        }
      }
      const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(vertices,3));geo.setIndex(indices);geo.computeVertexNormals();
      const mat=surface(band?'#809891':'#657e6d');mat.side=T.DoubleSide;this.scene.add(this.mesh(geo,mat));
    }
    const poleMat=surface('#e9e7cf');
    for(const [i,pin] of this.rangePins.entries()){
      const pole=this.mesh(new T.CylinderGeometry(.038,.048,2.6,8),poleMat);
      pole.position.set(pin.x,1.24,pin.z);this.scene.add(pole);
      const geo=new T.BufferGeometry(),vs=[],indices=[];
      for(let j=0;j<=8;j++){
        const u=j/8,fold=Math.sin(u*7+i)*u*.15;
        vs.push(1.05*u,2.52-u*.11,fold,1.05*u,2.04+u*.32,fold);
        if(j<8){const k=j*2;indices.push(k,k+1,k+2,k+1,k+3,k+2);}
      }
      geo.setAttribute('position',new T.Float32BufferAttribute(vs,3));geo.setIndex(indices);geo.computeVertexNormals();
      const flagMat=surface(i%2?'#c58740':'#b64532');flagMat.side=T.DoubleSide;
      const flag=this.mesh(geo,flagMat);flag.position.set(pin.x,0,pin.z);this.scene.add(flag);
    }
    this.buildWoods(random);
    this.buildHittingBay(random);
  }
  buildWoods(random){
    const T=THREE,wood=[],leaves=[],trees=[];
    const branch=(a,b,r)=>{
      const d=b.clone().sub(a);
      wood.push({p:a.clone().lerp(b,.5),q:new T.Quaternion().setFromUnitVectors(new T.Vector3(0,1,0),d.clone().normalize()),
        scale:new T.Vector3(r,d.length(),r),color:'#645440'});
    };
    const foliage=['#344f38','#43613d','#536c43','#637a49','#727f49','#4e6541'];
    for(let i=0;i<108;i++){
      const far=i>83;
      const x=far?random()*150-75:(i%2?1:-1)*(44+random()*35);
      const z=far?-238-random()*24:-8-random()*216;
      const h=5+random()*7,r=h*(.22+random()*.12);
      const foot=new T.Vector3(x,-.18,z),fork=new T.Vector3(x+.15,h*.45,z);
      trees.push({x,z,h,r});
      branch(foot,fork,h*.037);
      const tone=Math.floor(random()*foliage.length);
      for(let j=0;j<9;j++){
        const a=j*2.4+random()*.6;
        const width=j===0?0:r*(.38+random()*.5);
        const crown=new T.Vector3(x+Math.cos(a)*width,j===0?h:h*(.60+random()*.28),z+Math.sin(a)*width);
        if(j<6)branch(fork,crown,h*.016);
        leaves.push({p:crown,q:new T.Quaternion().setFromEuler(new T.Euler(random(),random()*6,random()*.45)),
          scale:new T.Vector3(r*(.61+random()*.26),r*(.65+random()*.40),r*(.63+random()*.33)),
          color:foliage[j%4===0?(tone+1)%foliage.length:tone]});
      }
    }
    if(typeof document!=='undefined'){
      const makeMap=(size,paint)=>{
        const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
        paint(canvas.getContext('2d'),size);
        const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;return map;
      };
      const shade=makeMap(64,(ctx,n)=>{
        const g=ctx.createRadialGradient(n/2,n/2,0,n/2,n/2,n/2);
        g.addColorStop(0,'rgba(26,43,30,.52)');g.addColorStop(.55,'rgba(26,43,30,.28)');g.addColorStop(1,'rgba(26,43,30,0)');
        ctx.fillStyle=g;ctx.fillRect(0,0,n,n);
      });
      const shadows=new T.InstancedMesh(new T.PlaneGeometry(1,1),
        new T.MeshBasicMaterial({map:shade,transparent:true,depthWrite:false}),trees.length);
      const flat=new T.Quaternion().setFromAxisAngle(new T.Vector3(1,0,0),-Math.PI/2);
      trees.forEach(({x,z,h,r},i)=>shadows.setMatrixAt(i,new T.Matrix4().compose(
        new T.Vector3(x+h*.13,-.051,z-h*.1),flat,new T.Vector3(r*3.5,r*2.8,1))));
      this.scene.add(shadows);
      // Crossed sprays give each crown a broken leafy silhouette at eye level and
      // from the aerial camera. A smaller solid core keeps the interior in shade.
      const foliageMap=makeMap(128,(ctx)=>{
        for(let j=0;j<360;j++){
          const a=random()*2*Math.PI,d=Math.sqrt(random()),r=(.79+.16*Math.sin(5*a)+.08*Math.cos(9*a))*d;
          const x=64+54*r*Math.cos(a),y=64+55*r*Math.sin(a);
          const value=Math.round(135+random()*64+(1-y/128)*48);
          ctx.fillStyle='rgb('+value+','+value+','+value+')';
          ctx.beginPath();ctx.ellipse(x,y,2+random()*7,1.5+random()*4,a,0,Math.PI*2);ctx.fill();
        }
      });
      const sprays=new T.InstancedMesh(new T.PlaneGeometry(2,2),
        new T.MeshBasicMaterial({map:foliageMap,alphaTest:.38,side:T.DoubleSide}),leaves.length);
      leaves.forEach((v,i)=>{
        const rot=new T.Quaternion().setFromEuler(new T.Euler(i%3===0 ? .94 : .15,i*2.4,Math.sin(i)*.21,'YXZ'));
        sprays.setMatrixAt(i,new T.Matrix4().compose(v.p,rot,v.scale.clone().multiplyScalar(1.45)));
        sprays.setColorAt(i,new T.Color(v.color).multiplyScalar(1.65));
        v.scale.multiplyScalar(.79);
      });
      this.scene.add(sprays);
    }
    for(const [geometry,items] of [[new T.CylinderGeometry(.55,1,1,7),wood],[new T.IcosahedronGeometry(1,1),leaves]]){
      const batch=new T.InstancedMesh(geometry,new T.MeshStandardMaterial({roughness:1}),items.length);
      items.forEach((v,i)=>{batch.setMatrixAt(i,new T.Matrix4().compose(v.p,v.q,v.scale));batch.setColorAt(i,new T.Color(v.color));});
      this.scene.add(batch);
    }
  }
  buildHittingBay(random){
    const T=THREE,steel=new T.MeshStandardMaterial({color:'#374642',metalness:.38,roughness:.55});
    const rubber=this.mat('#28332e'),balls=this.mat('#f4f0dd',55);
    if(typeof document!=='undefined'){
      const fibers=document.createElement('canvas');fibers.width=fibers.height=128;
      const ctx=fibers.getContext('2d');ctx.fillStyle='#e2e5db';ctx.fillRect(0,0,128,128);
      for(let i=0;i<4800;i++){
        ctx.fillStyle=i%2?'rgba(64,78,60,.34)':'rgba(252,255,239,.7)';
        ctx.fillRect(random()*128,random()*128,1,1+random()*3);
      }
      const nap=new T.CanvasTexture(fibers);nap.wrapS=nap.wrapT=T.RepeatWrapping;
      nap.repeat.set(6,6);nap.colorSpace=T.SRGBColorSpace;
      this.materials.turfMat.map=nap;this.materials.hittingStrip.map=nap;
    }
    this.box([12,.09,3.65],[0,-.04,.7],'#978e79');
    for(const x of [-3.1,0,3.1]){
      this.box([2.4,.034,2.48],[x,.014,.2],rubber);
      this.box([2.34,.021,2.42],[x,.035,.2],this.materials.turfMat);
      const stripThickness=.006;
      this.box([.40,stripThickness,1.42],[x+this.ballOrigin.x,this.hittingStripTop-stripThickness/2,this.ballOrigin.z],this.materials.hittingStrip);
      this.box([.67,.012,.035],[x-.25,.051,-.90],'#c5c5ac');
    }
    const rod=(a,b,r,mat)=>{
      const mesh=this.mesh(new T.CylinderGeometry(r,r,1,8),mat);this.segment(mesh,a,b);this.scene.add(mesh);
    };
    const bx=1.02,bz=.70;
    const hoop=(y,r)=>{
      const m=this.mesh(new T.TorusGeometry(r,.006,6,36),steel);
      m.rotation.x=Math.PI/2;m.position.set(bx,y,bz);this.scene.add(m);
    };
    for(const [y,r] of [[.055,.108],[.14,.132],[.221,.153],[.277,.168]])hoop(y,r);
    for(let i=0;i<18;i++){
      const a=i*Math.PI/9;
      rod(new T.Vector3(bx+.108*Math.cos(a),.055,bz+.108*Math.sin(a)),
          new T.Vector3(bx+.168*Math.cos(a+.12),.277,bz+.168*Math.sin(a+.12)),.0038,steel);
    }
    const handle=[];
    for(let i=0;i<=24;i++){
      const a=Math.PI*i/24;
      handle.push(new T.Vector3(bx+.166*Math.cos(a),.26+.17*Math.sin(a),bz+.045*Math.sin(a)));
    }
    this.scene.add(this.mesh(new T.TubeGeometry(new T.CatmullRomCurve3(handle),24,.005,6,false),steel));
    const centers=[];
    for(let layer=0;layer<6;layer++){
      const y=.078+layer*.035,limit=.083+.009*layer;
      for(let row=-4;row<=4;row++)for(let col=-4;col<=4;col++){
        const dx=(col+((row+layer)&1)*.5)*.044,dz=row*.038+(layer%2)*.014;
        if(Math.hypot(dx,dz)<=limit)centers.push([bx+dx,y+random()*.002,bz+dz]);
      }
    }
    this.box([.23,.018,.66],[.94,.055,.10],rubber);
    for(const side of [-1,1])this.box([.014,.035,.66],[.94+side*.11,.068,.10],steel);
    for(let i=0;i<7;i++)centers.push([.94+(i%2-.5)*.048,.086,.32-i*.059]);
    const batch=new T.InstancedMesh(new T.SphereGeometry(.02135,12,8),balls,centers.length);
    centers.forEach(([x,y,z],i)=>batch.setMatrixAt(i,new T.Matrix4().makeTranslation(x,y,z)));
    batch.castShadow=true;batch.receiveShadow=true;this.scene.add(batch);
  }
  buildGolfer(){const T=THREE;this.parts=[];
    const profile=(rings,material)=>{
      const geometry=new T.LatheGeometry(rings.map(([y,r])=>new T.Vector2(r,y)),32);
      return this.mesh(geometry,material);
    };
    const limb=(a,b,radii,material)=>{
      const mesh=profile(radii.map(([y,r])=>[y-.5,r]),material);
      this.scene.add(mesh);this.parts.push({a,b,mesh});
    };
    for(const [shoulder,elbow,wrist,hip,knee,ankle] of [[11,13,15,23,25,27],[12,14,16,24,26,28]]){
      limb(shoulder,elbow,[[-.08,.012],[0,.051],[.25,.054],[.67,.043],[.95,.037],[1.06,.025]],this.materials.skin);
      limb(shoulder,elbow,[[-.09,.018],[0,.057],[.17,.065],[.46,.061],[.57,.055],[.58,.047]],this.materials.shirt);
      limb(elbow,wrist,[[-.10,.018],[0,.037],[.22,.043],[.47,.037],[.85,.025],[1.03,.023]],this.materials.skin);

    }
    this.torso=new T.Group();this.scene.add(this.torso);
    this.hips=new T.Group();this.scene.add(this.hips);
    // A pair of trouser tubes joins one pelvis ring. The shirt shares the
    // waistband vertices and gradually takes on the rotation of the shoulders.
    const cloth=new T.BufferGeometry(),tri=[];
    const n=24,legRows=10,bodyRows=9,bodyStart=2*n*legRows;
    cloth.setAttribute('position',new T.Float32BufferAttribute(new Float32Array((bodyStart+n*bodyRows)*3),3));
    const leg=(side,r,j)=>(side*legRows+r)*n+(j+n)%n;
    const trunk=(r,j)=>bodyStart+r*n+(j+n)%n;
    const quad=(a,b,c,d)=>tri.push(a,b,c,b,d,c);
    for(let side=0;side<2;side++){
      const step=side===0?-1:1;
      for(let r=0;r<legRows-1;r++)for(let j=0;j<n;j++)
        quad(leg(side,r,j),leg(side,r+1,j),leg(side,r,j+step),leg(side,r+1,j+step));
      for(let j=-6;j<6;j++){
        const k=side===0?12-j:j;
        const a=leg(side,9,j),b=leg(side,9,j+1),c=trunk(0,k),d=trunk(0,k+step);
        if(side===0)quad(b,d,a,c);else quad(a,c,b,d);
      }
    }
    for(let r=0;r<bodyRows-1;r++){
      if(r===1)cloth.addGroup(0,tri.length,0);
      for(let j=0;j<n;j++)quad(trunk(r,j),trunk(r+1,j),trunk(r,j+1),trunk(r+1,j+1));
    }
    const pantsCount=cloth.groups[0].count;
    cloth.addGroup(pantsCount,tri.length-pantsCount,1);cloth.setIndex(tri);
    this.upperLegs=this.mesh(cloth,[this.materials.pants,this.materials.shirt]);
    this.upperLegs.frustumCulled=false;this.scene.add(this.upperLegs);
    this.neck=this.mesh(new T.CylinderGeometry(.041,.053,1,16),this.materials.skin);this.scene.add(this.neck);
    this.head=new T.Group();
    const oval=(parent,material,xyz,scale)=>{
      const part=this.mesh(new T.SphereGeometry(1,24,16),material);
      part.position.set(...xyz);part.scale.set(...scale);parent.add(part);return part;
    };
    const face=profile([[-.104,.034],[-.09,.052],[-.065,.064],[-.028,.075],[.018,.078],[.060,.072],[.089,.055],[.10,.01]],this.materials.skin);
    face.scale.z=1.13;this.head.add(face);
    oval(this.head,this.materials.skin,[0,-.026,-.082],[.015,.022,.019]);
    for(const side of [-1,1]){
      oval(this.head,this.materials.skin,[side*.076,-.017,.004],[.012,.024,.012]);
      oval(this.head,this.mat('#483d38',12),[side*.029,-.002,-.078],[.009,.003,.003]);
    }
    const crown=profile([[.017,.081],[.035,.085],[.062,.082],[.083,.072],[.100,.045],[.105,.004]],this.materials.cap);
    crown.scale.z=1.12;this.head.add(crown);
    const visor=oval(this.head,this.materials.cap,[0,.020,-.080],[.081,.007,.068]);visor.rotation.x=-.12;
    this.scene.add(this.head);
    this.shoes=[];
    for(let k=0;k<2;k++){
      const shoe=new T.Group();this.scene.add(shoe);this.shoes.push(shoe);
      // Heel, instep, ball of foot and tapered toe share one shaped upper.
      const stations=[[-.071,.034,.055],[-.054,.049,.086],[-.016,.054,.094],[.036,.053,.080],[.091,.057,.059],[.145,.055,.042],[.18,.037,.030],[.194,.004,.017]];
      const verts=[],tri=[],sole=[];
      for(const [z,w,h] of stations){
        for(let i=0;i<20;i++){
          const a=i*2*Math.PI/20,dy=Math.sin(a);
          verts.push(w*Math.cos(a),-.034+(dy>0?h*dy:.006*dy),z);
          sole.push(w*1.025*Math.cos(a),-.033+(dy>0?.006:.011)*dy,z);
        }
      }
      for(let j=0;j<stations.length-1;j++)for(let i=0;i<20;i++){
        const a=j*20+i,b=j*20+(i+1)%20;tri.push(a,b,a+20,b,b+20,a+20);
      }
      for(const [points,mat] of [[verts,this.materials.shoe],[sole,this.materials.shoeTrim]]){
        const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(points,3));g.setIndex(tri);g.computeVertexNormals();shoe.add(this.mesh(g,mat));
      }
      for(let j=0;j<4;j++){
        const lace=this.mesh(new T.CapsuleGeometry(.0025,.045,3,8),this.materials.glove);
        lace.rotation.z=Math.PI/2+(j%2?.12:-.12);lace.position.set(0,.051-j*.005,.015+j*.016);shoe.add(lace);
      }
    }
    // Hands: lead hand in white golf glove, trail hand in skin tone
    this.hands=[];for(let i=0;i<2;i++){const hand=this.mesh(new T.SphereGeometry(.034,14,10),i===0?this.materials.glove:this.materials.skin);this.scene.add(hand);this.hands.push(hand)}

    // Realistic Forged 7-Iron
    this.shaft=this.mesh(new T.CylinderGeometry(.0055,.0075,1,16),this.materials.shaft);this.scene.add(this.shaft);
    this.grip=this.mesh(new T.CylinderGeometry(.0125,.0105,1,16),this.materials.grip);this.scene.add(this.grip);

    // Chrome hosel, black ferrule, and forged blade with 34-degree loft
    this.clubhead=new T.Group();
    const ferrule=this.mesh(new T.CylinderGeometry(.0072,.0072,.022,12),this.materials.ferrule);ferrule.position.set(0,.036,0);this.clubhead.add(ferrule);
    const hosel=this.mesh(new T.CylinderGeometry(.0065,.0065,.042,12),this.materials.shaft);hosel.position.set(0,.016,0);this.clubhead.add(hosel);
    const bladeBody=this.mesh(new T.BoxGeometry(.086,.038,.024),this.materials.ironBody);bladeBody.position.set(.038,-.012,-.005);this.clubhead.add(bladeBody);
    const facePlate=this.mesh(new T.BoxGeometry(.084,.040,.008),this.materials.ironFace);
    facePlate.position.set(.038,-.01,-.014);facePlate.rotation.x=34*Math.PI/180;
    this.clubhead.add(facePlate);this.facePlate=facePlate;
    this.scene.add(this.clubhead);
  }
  segment(mesh,a,b){const d=b.clone().sub(a);mesh.position.copy(a).add(b).multiplyScalar(.5);mesh.scale.y=Math.max(.001,d.length());mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),d.normalize())}
  poseAt(row,frame){const frames=row.avatar_motion?.frames||row.world;const i=Math.max(0,Math.min(frames.length-1,frame)),lo=Math.floor(i),hi=Math.min(frames.length-1,lo+1),u=i-lo;const a=frames[lo],b=frames[hi];return a.map((v,j)=>new THREE.Vector3((v[0]||0)*(1-u)+(b[j][0]||0)*u,-((v[1]||0)*(1-u)+(b[j][1]||0)*u),-((v[2]||0)*(1-u)+(b[j][2]||0)*u)))}
  registerStance(row){
    const open=this.poseAt(row,0);
    const length=row.club_display?.club_length_display_units||.94615;
    const mid=open[27].clone().lerp(open[28],.5);
    this.stanceOffset=new THREE.Vector3(this.ballOrigin.x-length*.53-.195-Math.max(open[27].x,open[28].x),0,this.ballOrigin.z-mid.z);
    this.shoeYaw=[0,1].map(k=>{
      const direction=open[31+k].clone().sub(open[27+k]);
      return Math.PI/2-THREE.MathUtils.clamp(Math.atan2(direction.z,Math.abs(direction.x)),-.45,.45);
    });
    // Centered filtering is deterministic when scrubbing in either direction.
    const frames=row.avatar_motion?.frames||row.world;
    this.finishLegs=frames.map((_,i)=>Array.from({length:4},(_,j)=>
      [0,1,2].map(axis=>{
        const values=[];
        for(let t=-3;t<=3;t++)values.push(frames[Math.max(0,Math.min(frames.length-1,i+t))][25+j][axis]);
        values.sort((a,b)=>a-b);
        return (values[2]+2*values[3]+values[4])/4;
      })));
    this.addressAnkles=open.slice(27,29).map(v=>v.clone().add(this.stanceOffset));
    this.addressAnkleHeight=open.slice(27,29).map(point=>point.y-Math.min(open[27].y,open[28].y));
    const grip=open[15].clone().lerp(open[16],.5);
    this.takeaway=Math.max(0,this.impact-.65);
    let consecutive=0;
    for(let i=1;i<row.world.length && i/row.fps<this.impact;i++){
      const pose=this.poseAt(row,i);
      const moved=pose[15].clone().lerp(pose[16],.5).distanceTo(grip)>.055;
      consecutive=moved?consecutive+1:0;
      if(consecutive===3){
        this.takeaway=Math.min(this.takeaway,(i-2)/row.fps);break;
      }
    }
  }
  squareIron(direction){
    const up=direction.clone().negate();
    const toe=new THREE.Vector3(1,0,0).addScaledVector(up,-up.x).normalize();
    const back=toe.clone().cross(up);
    return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(toe,up,back));
  }
  setupShot(row){
    this.lastShot=row.id;this.noPose=row.prediction_status==='abstained_insufficient_pose';
    for(const part of this.parts)part.mesh.visible=!this.noPose;
    for(const obj of [this.upperLegs,this.torso,this.neck,this.hips,this.head,...this.shoes,...this.hands,this.shaft,this.grip,this.clubhead,this.ballTruth,this.traceTruth,this.ballPred,this.tracePred,this.ribbonTruth,this.ribbonPred,this.landRingTruth,this.landRingPred,this.discTruth,this.discPred,this.outerRingTruth,this.outerRingPred,this.totalRingTruth,this.totalRingPred])obj.visible=!this.noPose;
    
    const n=row.world.length;let floor=[];
    for(let i=0;i<n;i++){const p=this.poseAt(row,i);floor.push(Math.min(p[27].y,p[28].y))}
    floor.sort((a,b)=>a-b);this.lift=-floor[Math.floor(floor.length*.25)]+.09;
    const impact=row.departure_interval?(row.departure_interval.lower_s+row.departure_interval.upper_s)/2-row.start:Math.min((n-1)/row.fps-.4,2.0);
    this.poseDuration=(n-1)/row.fps;
    this.impact=Math.min(this.poseDuration-.01,Math.max(.01,impact));
    this.launchTime=this.impact;
    
    this.registerStance(row);

    // Legacy metrics maintained for test assertions
    this.distance=12+Math.max(0,Math.min(200,row.prediction??100))*.16;
    const trackedAngle=row.metrics?.ball_angle_deg??row.ball_angle_deg??row.launch_direction??0;
    this.lateral=Math.max(-8,Math.min(8,Math.tan(trackedAngle*Math.PI/180)*this.distance));

    // 1. TrackMan Truth flight & rollout parameters
    this.hasTruth=Number.isFinite(row.carry);
    this.carryTruth=Number.isFinite(row.carry)?Math.max(0,row.carry):100;
    this.totalTruth=Number.isFinite(row.total_actual)?row.total_actual:(this.carryTruth+8.5);
    this.distTruth=this.carryTruth*.9144;
    this.rollTruth=(this.totalTruth-this.carryTruth)*.9144;
    this.dirTruth=Number.isFinite(row.launch_direction)?row.launch_direction:0;
    this.sideTruth=Number.isFinite(row.carry_side_ft)?(row.carry_side_ft*.3048):(Math.tan(this.dirTruth*Math.PI/180)*this.distTruth);
    this.heightTruth=Number.isFinite(row.height_ft)?(row.height_ft*.3048):(3.5+this.distTruth*.12);

    // 2. Best Model Predicted flight & rollout parameters
    this.hasPred=Number.isFinite(row.best_prediction);
    this.carryPred=Number.isFinite(row.best_prediction)?Math.max(0,row.best_prediction):(Number.isFinite(row.prediction)?Math.max(0,row.prediction):100);
    this.totalPred=Number.isFinite(row.best_total_yd)?row.best_total_yd:(this.carryPred+8.5);
    this.distPred=this.carryPred*.9144;
    this.rollPred=(this.totalPred-this.carryPred)*.9144;
    this.dirPred=Number.isFinite(row.best_launch_direction)?row.best_launch_direction:0;
    this.sidePred=Number.isFinite(row.best_carry_side_ft)?(row.best_carry_side_ft*.3048):(Math.tan(this.dirPred*Math.PI/180)*this.distPred);
    this.heightPred=Number.isFinite(row.best_height_ft)?(row.best_height_ft*.3048):(3.5+this.distPred*.12);
    this.curvePred=Number.isFinite(row.best_curve_ft)?(row.best_curve_ft*.3048):null;
    this.flights = [
      this.makeFlight('Truth', row.hang_time_s,row.total_side_ft),
      this.makeFlight('Pred', row.best_hang_time,row.best_total_side_ft)
    ].filter(Boolean);
    // Frame the whole shot from address, including the farther rollout.
    this.overviewPoints=[new THREE.Vector3(-1,0,1),new THREE.Vector3(1,2.6,-1)];
    this.firstLanding=Math.min(...this.flights.map(f=>f.hang));
    this.markerLayout=null;
    const finishBounds=new THREE.Box3();
    for(const flight of this.flights){
      flight.trail=this.sampleTrail(flight);
      for(let i=0;i<=24;i++)this.overviewPoints.push(this.flightPoint(flight,i/24));
      const land=this.flightPoint(flight,1);
      for(const [prefix,point] of [['outerRing',land],['totalRing',land.clone().add(flight.endpoint)]]){
        const radius=this[prefix+flight.key].geometry.parameters.outerRadius;
        const extent=new THREE.Vector3(radius,0,radius);
        finishBounds.expandByPoint(point.clone().sub(extent));
        finishBounds.expandByPoint(point.clone().add(extent));
      }
      for(const sample of flight.trail){
        if(sample.time>=this.firstLanding)finishBounds.expandByPoint(sample.point);
        if(sample.time>flight.hang)this.overviewPoints.push(sample.point);
      }
      finishBounds.expandByPoint(this.flightPoint(flight,this.firstLanding/flight.hang));
    }
    this.finishView=finishBounds.getBoundingSphere(new THREE.Sphere());
    this.finishView.radius=Math.max(2.35,this.finishView.radius*1.12+1);
    this.overviewCenter=new THREE.Box3().setFromPoints(this.overviewPoints).getCenter(new THREE.Vector3());
    this.postDuration = Math.max(0,...this.flights.map(f=>f.hang+f.rollTime));
    this.endTime = Math.max(this.poseDuration,this.launchTime+this.postDuration);
  }
  makeFlight(key, hang, finalSideFt){
    if(!this['has'+key])return null;
    const height=Math.max(0,this['height'+key]);
    const estimated=!(Number.isFinite(hang)&&hang>0);
    if(estimated)hang=Math.max(.15,Math.sqrt(8*height/9.81));
    const downrange=this['roll'+key],dist=this['dist'+key],side=this['side'+key];
    const dir=this['dir'+key]*Math.PI/180;
    // The derivative of the airborne x(u) at u=1 gives the landing heading.
    // Spend the carry-to-total distance along that heading, including on severe misses.
    const rollVector=new THREE.Vector3(dist>0?2*side/dist-Math.tan(dir):0,0,-1).normalize();
    const endpoint=Number.isFinite(finalSideFt)
      ?new THREE.Vector3(finalSideFt*.3048-side,0,-downrange)
      :rollVector.clone().multiplyScalar(downrange);
    const roll=endpoint.length();
    const hopHeight=Math.min(.85,height*.024);
    const hopTime=Math.sqrt(8*hopHeight/9.81);
    const groundTime=Math.sqrt(2*Math.abs(roll)/2.4);
    const rollTime=hopTime+groundTime;
    const hopDistance=roll>=0 ? roll*hopTime/(hopTime+groundTime/2||1) : Math.min(.45,-roll*.08);
    return {key,height,hang,estimated,roll,rollTime,dist,side,dir,rollVector,endpoint,downrange,
      hopHeight,hopTime,hopDistance,groundTime};
  }
  sourceToClock(t){
    if(t<=this.impact)return t;
    return this.launchTime+Math.min(1,(t-this.impact)/(this.poseDuration-this.impact))*(this.endTime-this.launchTime);
  }
  flightPoint(f,q){
    const u=q*(1.22-.22*q);
    const x=Math.tan(f.dir)*f.dist*u*(1-u)+f.side*u*u;
    const ground=this.ballGroundY(this.ballOrigin.z-f.dist);
    return new THREE.Vector3(this.ballOrigin.x+x,
      this.ballOrigin.y*(1-q)+ground*q+4*f.height*q*(1-q),this.ballOrigin.z-f.dist*u);
  }
  ballGroundY(z){
    // The mowing pattern is painted on one continuous turf surface.
    return this.ballRadius-.06;
  }
  groundState(f,elapsed){
    const land=this.flightPoint(f,1);
    const h=f.hopTime?Math.min(1,elapsed/f.hopTime):1;
    const g=f.groundTime?Math.max(0,Math.min(1,(elapsed-f.hopTime)/f.groundTime)):0;
    // Hermite rollout begins with the hop's horizontal velocity and stops at total.
    const startSpeed=f.hopTime?f.hopDistance/f.hopTime:2*f.roll/(f.groundTime||1);
    let run=f.hopDistance*h+(f.roll-f.hopDistance)*g*g*(3-2*g)
      +startSpeed*f.groundTime*g*(1-g)*(1-g);
    if(elapsed+1e-9>=f.rollTime)run=f.roll;
    const pos=land.clone().addScaledVector(f.rollVector,run);
    const progress=f.roll?run/f.roll:1;
    // Ease from the incoming heading to the estimated two-dimensional endpoint.
    pos.addScaledVector(f.endpoint.clone().addScaledVector(f.rollVector,-f.roll),progress*progress);
    pos.y=this.ballGroundY(pos.z)+4*f.hopHeight*h*(1-h);
    return {pos,run};
  }
  sampleTrail(f){
    const samples=[];
    for(let i=0;i<=this.trailSteps.air;i++){
      const q=i/this.trailSteps.air;
      samples.push({time:q*f.hang,point:this.flightPoint(f,q)});
    }
    let start=0;
    for(const [duration,steps] of [[f.hopTime,this.trailSteps.hop],[f.groundTime,this.trailSteps.roll]]){
      if(duration>0){
        for(let i=1;i<=steps;i++){
          const elapsed=start+duration*i/steps;
          samples.push({time:f.hang+elapsed,point:this.groundState(f,elapsed).pos});
        }
      }
      start+=duration;
    }
    return samples;
  }
  updateFlight(f,seconds){
    const key=f.key,q=Math.min(1,seconds/f.hang),landed=seconds+1e-9>=f.hang;
    const elapsed=Math.max(0,seconds-f.hang);
    const r=landed?(f.rollTime?Math.min(1,(elapsed+1e-9)/f.rollTime):1):0;
    const land=this.flightPoint(f,1);
    const {pos,run}=landed?this.groundState(f,elapsed):{pos:this.flightPoint(f,q),run:0};
    const ball=this['ball'+key];
    ball.visible=true;ball.position.copy(pos);
    this['landRing'+key].position.set(land.x,.016,land.z);
    for(const prefix of ['landRing','disc','outerRing']){
      const obj=this[prefix+key];obj.visible=landed;obj.position.set(land.x,obj.position.y||.016,land.z);
    }
    const total=this['totalRing'+key];total.visible=landed&&r===1;
    total.position.copy(land).add(f.endpoint).y=.018;
    f.q=q;f.r=r;f.yards=this['carry'+key]*q*(1.22-.22*q)+(f.roll?run/f.roll:0)*f.downrange/.9144;
  }
  drawMarkers(){
    const canvas=this.markerOverlay;
    if(!canvas)return;
    const w=this.width,h=this.height,dpr=Math.min(window.devicePixelRatio||1,2);
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){
      canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);
    }
    const ctx=canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
    const project=p=>{
      const v=p.clone().project(this.camera);
      return {x:(v.x+1)*w/2,y:(1-v.y)*h/2,z:v.z};
    };
    if(!this.flights.some(f=>this['landRing'+f.key].visible)){
      this.drawPinYardages(ctx,project,w,h,[]);return;
    }
    const signature=[w,h,...this.camera.projectionMatrix.elements,...this.camera.matrixWorld.elements].join('|');
    if(this.markerLayout?.signature!==signature){
      this.markerLayout={signature,labels:this.planMarkers(ctx,project,w,h)};
    }
    const visible=this.markerLayout.labels.filter(label=>label.marker.visible);
    this.drawPinYardages(ctx,project,w,h,this.markerLayout.labels);
    for(const {anchor,x,y,width,color} of visible){
      ctx.strokeStyle=color;ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(anchor.x,anchor.y);
      ctx.lineTo(Math.max(x,Math.min(x+width,anchor.x)),Math.max(y,Math.min(y+19,anchor.y)));ctx.stroke();
    }
    for(const {x,y,width,color,text} of visible){
      ctx.fillStyle='rgba(12,26,32,.92)';ctx.fillRect(x,y,width,19);
      ctx.fillStyle=color;ctx.fillText(text,x+6,y+13);
    }
  }
  drawPinYardages(ctx,project,w,h,labels){
    const occupied=labels.map(l=>({x:l.x,y:l.y,w:l.width,h:19}));
    const protect=(p,r)=>{
      const v=project(p);
      if(Math.abs(v.z)<=1)occupied.push({x:v.x-r,y:v.y-r,w:2*r,h:2*r});
    };
    protect(this.head.position,14);
    const trailSegments=[];
    for(const f of this.flights){
      protect(this['ball'+f.key].position,8);
      let previous;
      for(const point of f.visibleTrail){
        const current=project(point);
        if(previous && Math.abs(previous.z)<=1 && Math.abs(current.z)<=1){
          trailSegments.push([previous,current]);
        }
        previous=current;
      }
    }
    ctx.font='600 10px system-ui';
    for(const pin of this.rangePins){
      const anchor=project(new THREE.Vector3(pin.x,2.52,pin.z));
      const text=pin.yd+' yd',width=Math.ceil(ctx.measureText(text).width)+12;
      if(Math.abs(anchor.z)>1||anchor.x<3||anchor.x>w-3||anchor.y<0||anchor.y>h)continue;
      const x=anchor.x-width/2,y=anchor.y-20;
      if(x<3||x+width>w-3||y<3||y+17>h-3)continue;
      if(occupied.some(b=>x<b.x+b.w+3 && x+width>b.x-3 && y<b.y+b.h+3 && y+17>b.y-3))continue;
      if(trailSegments.some(([a,b])=>{
        const halfWidth=width/2+5,halfHeight=12.5;
        const centerX=anchor.x,centerY=y+7.5;
        if(Math.max(a.x,b.x)<centerX-halfWidth||Math.min(a.x,b.x)>centerX+halfWidth||
           Math.max(a.y,b.y)<centerY-halfHeight||Math.min(a.y,b.y)>centerY+halfHeight)return false;
        // After the screen axes, test separation on the segment's normal.
        const dx=b.x-a.x,dy=b.y-a.y;
        return Math.abs(dy*(centerX-a.x)-dx*(centerY-a.y))<=
          Math.abs(dy)*halfWidth+Math.abs(dx)*halfHeight;
      }))continue;
      ctx.fillStyle='rgba(30,52,42,.84)';ctx.fillRect(x,y,width,15);
      ctx.fillStyle='#f9efd6';ctx.fillText(text,x+6,y+11);
      ctx.strokeStyle='#e4dabc';ctx.beginPath();ctx.moveTo(anchor.x,y+15);ctx.lineTo(anchor.x,anchor.y);ctx.stroke();
      occupied.push({x,y,w:width,h:17});
    }
    ctx.font='600 11px system-ui';
  }
  planMarkers(ctx,project,w,h){
    const labels=[],path=[];
    ctx.font='600 11px system-ui';
    for(const f of this.flights){
      for(let i=0;i<f.trail.length;i+=2)path.push(project(f.trail[i].point));
      for(const [kind,prefix,metric] of [['Carry','landRing','carry'],['Total','totalRing','total']]){
        const marker=this[prefix+f.key];
        const anchor=project(marker.position);
        if(Math.abs(anchor.z)>1||anchor.x<0||anchor.x>w||anchor.y<0||anchor.y>h)continue;
        const text=kind+' '+Math.round(this[metric+f.key])+' yd';
        labels.push({anchor,marker,text,width:Math.ceil(ctx.measureText(text).width)+12,
          color:f.key==='Truth'?'#76ddff':'#dfb9ff'});
      }
    }
    const placed=[];
    for(const label of labels){
      const {anchor,width}=label;
      let best,lowest=Infinity;
      for(const dy of [-26,-49,7,-72,30,53]){
        for(const dx of [0,-width*.68,width*.68,-width,width]){
          const x=Math.max(2,Math.min(w-width-2,anchor.x-width/2+dx));
          const y=Math.max(2,Math.min(h-21,anchor.y+dy));
          const contains=p=>p.x>x-3&&p.x<x+width+3&&p.y>y-3&&p.y<y+22;
          const collisions=placed.filter(p=>p.x<x+width+3&&p.x+p.width>x-3&&p.y<y+22&&p.y+22>y).length;
          const cost=collisions*1e6+labels.filter(l=>contains(l.anchor)).length*5000+
            path.filter(contains).length*100+Math.hypot(x+width/2-anchor.x,y+19-anchor.y)+(dy>0?12:0);
          if(cost<lowest){lowest=cost;best={x,y};}
        }
      }
      placed.push({...label,...best});
    }
    return placed;
  }
  frameTee(offset){
    const T=THREE;
    const signature=[this.lastShot,this.width,this.height,this.dragYaw,this.dragPitch].join('|');
    if(this.teeFit?.signature!==signature){
      const tee=new T.Vector3(.15,1.05,0);
      const forward=new T.Vector3(-offset.x,0,-offset.z).normalize();
      const right=forward.clone().cross(new T.Vector3(0,1,0));
      const slope=Math.tan(this.camera.fov*Math.PI/360);
      let position,aim;
      // Try the closest spectator position first. Look up to include high flights
      // before backing away for exceptionally wide misses or narrow screens.
      for(let depth=5.4;depth<650;depth*=1.18){
        position=tee.clone().addScaledVector(forward,-depth);
        position.y=1.55+this.dragPitch*1.5;
        let low=Infinity,high=-Infinity,left=Infinity,rightmost=-Infinity;
        const rays=this.overviewPoints.map(point=>point.clone().sub(position));
        for(const ray of rays){
          const z=ray.dot(forward);
          const theta=Math.atan2(ray.dot(right),z),phi=Math.atan2(ray.y,z);
          left=Math.min(left,theta);rightmost=Math.max(rightmost,theta);
          low=Math.min(low,phi);high=Math.max(high,phi);
        }
        const az=(left+rightmost)/2,pitch=(low+high)/2;
        const direction=forward.clone().multiplyScalar(Math.cos(az)).addScaledVector(right,Math.sin(az));
        direction.multiplyScalar(Math.cos(pitch)).y=Math.sin(pitch);
        const side=direction.clone().cross(new T.Vector3(0,1,0)).normalize();
        const up=side.clone().cross(direction);
        aim=position.clone().add(direction);
        if(rays.every(ray=>{
          const limit=ray.dot(direction)*slope*.83;
          return Math.abs(ray.dot(up))<limit && Math.abs(ray.dot(side))<limit*this.camera.aspect;
        }))break;
      }
      this.teeFit={signature,position,aim};
    }
    this.camera.position.copy(this.teeFit.position);
    this.camera.lookAt(this.teeFit.aim);this.camera.updateMatrixWorld();
  }
  frameCamera(){
    const T=THREE,mode=this.cameraMode;
    const w=this.canvas.clientWidth||540,h=this.canvas.clientHeight||470;
    if(this.width!==w||this.height!==h){
      this.renderer.setSize(w,h,false);this.width=w;this.height=h;
      this.camera.aspect=w/h;this.updateCameraProjection();
    }
    const views={side_wide:[.047,.216,1],
      range:[.12,.09,1.45],landing:[-.55,.34,1.55],follow:[.24,.38,1.45]};
    const [x,y,z]=views[mode]||views.side_wide;
    const followBlend=mode==='follow'?T.MathUtils.smoothstep(this.airSeconds,0,.55):0;
    const overview=mode==='side_wide';
    const fov=overview?37:48;
    if(this.camera.fov!==fov){this.camera.fov=fov;this.camera.updateProjectionMatrix();}
    const baseElevation=Math.atan2(y,Math.hypot(x,z));
    const offset=new T.Vector3(x,overview?0:y+this.dragPitch,z).normalize();
    if(overview){
      const elevation=baseElevation+this.dragPitch;
      offset.multiplyScalar(Math.cos(elevation)).y=Math.sin(elevation);
    }
    offset.applyAxisAngle(new T.Vector3(0,1,0),this.dragYaw);
    if(mode==='range'){
      this.scene.fog.near=85;this.scene.fog.far=390;
      if(this.camera.far!==850){this.camera.far=850;this.camera.updateProjectionMatrix();}
      this.frameTee(offset);return;
    }
    const tee=new T.Vector3(.15,.97,0);
    let target=tee.clone(),radius=mode==='follow'?2.35:1.65;
    const halfFov=Math.atan(Math.tan(this.camera.fov*Math.PI/360)*Math.min(1,this.camera.aspect));
    let followDistance;
    if(this.flights.length && (mode==='follow'||mode==='landing')){
      const points=this.flights.map(f=>this['ball'+f.key].position);
      const middle=points.reduce((v,p)=>v.add(p),new T.Vector3()).divideScalar(points.length);
      const u=Math.min(1,this.airSeconds/(mode==='landing'?1.6:.23));
      const blend=u*u*(3-2*u);
      if(mode==='landing'){
        const landing=this.flights.map(f=>this.flightPoint(f,1));
        middle.copy(landing[0]);if(landing.length===2)middle.lerp(landing[1],.5);
        target.lerp(middle,blend);
        radius=Math.max(radius,...landing.map(p=>blend*(middle.distanceTo(p)+3)));
      }else{
        target.lerp(middle,blend);
        const transition=Math.min(.85,this.firstLanding*.35);
        const u=Math.max(0,Math.min(1,1+(this.airSeconds-this.firstLanding)/transition));
        const finishBlend=u*u*(3-2*u);
        const baseDistance=T.MathUtils.lerp(radius/Math.sin(halfFov),13.5,followBlend);
        const finishDistance=this.finishView.radius/Math.sin(halfFov);
        target.lerp(this.finishView.center,finishBlend);
        followDistance=T.MathUtils.lerp(baseDistance,finishDistance,finishBlend);
        if(u<1){
          // Keep a consistent sense of scale. Pull back only if a ball is
          // approaching the edge of the view rather than for every extra yard.
          const right=new T.Vector3(0,1,0).cross(offset).normalize();
          const up=offset.clone().cross(right);
          const slope=Math.tan(this.camera.fov*Math.PI/360)*.83;
          for(const point of points){
            const rel=point.clone().sub(target);
            const fit=Math.max(Math.abs(rel.dot(right))/(slope*this.camera.aspect),Math.abs(rel.dot(up))/slope);
            followDistance=Math.max(followDistance,rel.dot(offset)+fit+.65);
          }
        }
      }
    }
    let distance=followDistance??radius/Math.sin(halfFov);
    if(mode==='side_wide'){
      target.copy(this.overviewCenter);
      target.x=this.ballOrigin.x+(target.x-this.ballOrigin.x)*.35;
      target.y*=.24;
      const right=new T.Vector3(0,1,0).cross(offset).normalize();
      const up=offset.clone().cross(right);
      const slope=Math.tan(this.camera.fov*Math.PI/360);
      for(const point of this.overviewPoints){
        const rel=point.clone().sub(target);
        const fit=Math.max(Math.abs(rel.dot(right))/(slope*this.camera.aspect),Math.abs(rel.dot(up))/slope);
        distance=Math.max(distance,rel.dot(offset)+1.15*fit+1);
      }
    }
    this.scene.fog.near=mode==='side_wide'?distance+35:65;
    this.scene.fog.far=mode==='side_wide'?distance+230:220;
    const far=Math.max(320,this.scene.fog.far+15);
    if(this.camera.far!==far){this.camera.far=far;this.camera.updateProjectionMatrix();}
    this.camera.position.copy(target).addScaledVector(offset,distance);
    if(overview && this.camera.position.y<1.35){
      // Stop at eye level and keep the drag in sync with the clamped position.
      this.camera.position.y=1.35;
      const horizontal=Math.hypot(this.camera.position.x-target.x,this.camera.position.z-target.z);
      this.dragPitch=Math.atan2(this.camera.position.y-target.y,horizontal)-baseElevation;
    }
    this.camera.lookAt(target);this.camera.updateMatrixWorld();
  }
  updateRibbon(geom, pts, width=0.20){
    if(pts.length<2){geom.setDrawRange(0,0);return;}
    const arr=geom.attributes.position.array;
    const T=THREE;
    const camPos=this.camera.position;
    for(let j=0;j<pts.length;j++){
      const curr=pts[j];
      const prev=pts[Math.max(0,j-1)];
      const next=pts[Math.min(pts.length-1,j+1)];
      const tangent=next.clone().sub(prev).normalize();
      
      // Vector from point to camera for true camera-facing billboard ribbon
      const toCam=camPos.clone().sub(curr).normalize();
      let norm=new T.Vector3().crossVectors(tangent, toCam).normalize();
      if(norm.lengthSq()<1e-4)norm=new T.Vector3(-tangent.z,0,tangent.x).normalize();
      
      // Sleek Toptracer width scaling: clean, crisp, not bulky
      const dist=curr.distanceTo(camPos);
      const aerial=this.cameraMode==='side_wide';
      const w=Math.max(.012,Math.min(aerial ? .72 : .32,width*(.06+dist*(aerial ? .018 : .009))));
      norm.multiplyScalar(w*0.5);
      
      const idx=j*6;
      arr[idx]   = curr.x - norm.x;
      arr[idx+1] = Math.max(this.ballGroundY(curr.z)-.024, curr.y - norm.y);
      arr[idx+2] = curr.z - norm.z;
      
      arr[idx+3] = curr.x + norm.x;
      arr[idx+4] = Math.max(this.ballGroundY(curr.z)-.024, curr.y + norm.y);
      arr[idx+5] = curr.z + norm.z;
    }
    geom.attributes.position.needsUpdate=true;
    geom.setDrawRange(0,(pts.length-1)*6);
  }
  update(row,time,yaw,independent=false){
    if(!this.ready)return;
    if(this.lastShot!==row.id)this.setupShot(row);
    const clock=independent?time:this.sourceToClock(time);
    const t=Math.min(this.poseDuration,time);
    const T=THREE,p=this.poseAt(row,t*row.fps);
    const release=T.MathUtils.smoothstep(t,this.impact,this.impact+.20);
    if(release>0){
      const frame=Math.min(this.finishLegs.length-1,t*row.fps),lo=Math.floor(frame),hi=Math.ceil(frame);
      for(let j=0;j<4;j++){
        const point=new T.Vector3().fromArray(this.finishLegs[lo][j])
          .lerp(new T.Vector3().fromArray(this.finishLegs[hi][j]),frame-lo);
        point.y*=-1;point.z*=-1;p[25+j].lerp(point,release);
      }
    }
    const groundLift=.09-Math.min(p[27].y,p[28].y);
    for(const point of p){point.add(this.stanceOffset);point.y+=groundLift;}
    for(let k=0;k<2;k++){
      const ankle=p[27+k],shoe=this.shoes[k];
      ankle.y=Math.max(.09,ankle.y-this.addressAnkleHeight[k]);
      const pivot=T.MathUtils.smoothstep(t,this.impact-.025,this.impact+.45);
      const pitch=k===1?.75*pivot:0;
      const raw=ankle.clone(),anchor=this.addressAnkles[k];
      // Favor the planted toe when monocular depth makes the feet converge.
      ankle.x=T.MathUtils.lerp(ankle.x,anchor.x,pivot*.82);
      ankle.z=T.MathUtils.lerp(ankle.z,anchor.z,pivot*.82);
      shoe.rotation.set(pitch,this.shoeYaw[k]+(k===1?.60:.08)*pivot,0,'YXZ');
      const toe=new T.Vector3(0,-.044,.175);
      const contact=toe.clone().applyAxisAngle(new T.Vector3(0,1,0),this.shoeYaw[k]);
      shoe.position.copy(ankle).add(contact).sub(toe.applyQuaternion(shoe.quaternion));
      shoe.position.y=Math.max(.09,shoe.position.y);
      ankle.copy(shoe.position);
      p[25+k].addScaledVector(ankle.clone().sub(raw),.45);

    }
    const club=row.club_display?.frames[Math.max(0,Math.min(row.world.length-1,Math.round(t*row.fps)))];
    this.clubStatus=club?club.status:'model_synthesized';
    if(row.club_display){
      const marks=row.club_display.frames;
      const index=T.MathUtils.clamp(t*row.fps,0,marks.length-1);
      const a=marks[Math.floor(index)]?.hand_shift||[0,0,0],b=marks[Math.ceil(index)]?.hand_shift||[0,0,0];
      const shift=new T.Vector3().fromArray(a).lerp(new T.Vector3().fromArray(b),index%1);
      shift.multiply(new T.Vector3(1,-1,-1));p[15].add(shift);p[16].add(shift);
    }
    
    const shoulders=p[11].clone().add(p[12]).multiplyScalar(.5),hips=p[23].clone().add(p[24]).multiplyScalar(.5);
    this.segment(this.torso,hips,shoulders);
    // Ears locate the cranium; the nose locates its forward direction, not its center.
    const earMid=p[7].clone().add(p[8]).multiplyScalar(.5),right=p[8].clone().sub(p[7]).normalize(),front=p[0].clone().sub(earMid);front.addScaledVector(right,-front.dot(right)).normalize();
    let up=right.clone().cross(front).normalize();
    const bodyUp=shoulders.clone().sub(hips).normalize();
    if(up.dot(bodyUp)<0){right.negate();up.negate()}
    // Blend noisy ear tilt toward the torso while preserving the nose direction.
    up.lerp(bodyUp,.58).normalize();
    front.addScaledVector(up,-front.dot(up)).normalize();
    right.copy(front).cross(up).normalize();
    if(right.lengthSq()>.5&&front.lengthSq()>.5&&up.lengthSq()>.5){this.head.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(right,up,front.clone().negate()));this.head.position.copy(earMid).addScaledVector(up,.025)}else{this.head.quaternion.identity();this.head.position.copy(shoulders).add(new T.Vector3(0,.2,0));up.set(0,1,0)}
    this.segment(this.neck,shoulders.clone().addScaledVector(shoulders.clone().sub(hips).normalize(),.035),this.head.position.clone().addScaledVector(up,-.09));
    const spine=shoulders.clone().sub(hips).normalize(),bodyRight=p[12].clone().sub(p[11]);bodyRight.addScaledVector(spine,-bodyRight.dot(spine)).normalize();const back=bodyRight.clone().cross(spine).normalize();this.torso.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(bodyRight,spine,back));const chest=T.MathUtils.clamp(p[11].distanceTo(p[12])*.52,.14,.22);
    this.torso.scale.x=chest;this.torso.scale.z=chest*.66;
    const hipAxis=p[24].clone().sub(p[23]).normalize();
    const hipBack=hipAxis.clone().cross(spine).normalize(),hipUp=hipBack.clone().cross(hipAxis);
    this.hips.quaternion.setFromRotationMatrix(new T.Matrix4().makeBasis(hipAxis,hipUp,hipBack));
    this.hips.position.copy(hips);
    const positions=this.upperLegs.geometry.attributes.position;
    for(let side=0;side<2;side++){
      const sign=side===0?-1:1,hip=p[23+side],knee=p[25+side];
      const span=p[23].distanceTo(p[24])*.5;
      const ankle=p[27+side];
      // Sample both sides of the kneecap on the same surface. A bisector
      // there rounds the bend while leaving the captured knee in place.
      const shin=knee.clone().sub(ankle).normalize(),thigh=hip.clone().sub(knee).normalize();
      const bisector=shin.clone().add(thigh).normalize();
      for(let ring=0;ring<10;ring++){
        const u=[0,.22,.62,.88,1,.10,.36,.66,.86,1][ring];
        const center=ring<=4?ankle.clone().lerp(knee,u):knee.clone().lerp(hip,u);
        const axis=ring<=2?shin:ring===3?shin.clone().lerp(bisector,.45):ring===4?bisector:ring===5?thigh.clone().lerp(bisector,.45):thigh;
        const depth=hipAxis.clone().cross(axis).normalize(),across=axis.clone().cross(depth).normalize();
        const width=[.043,.048,.060,.057,.057,.061,.073,.084,.084,.080][ring];
        for(let j=0;j<24;j++){
          const angle=j*Math.PI/12,c=Math.cos(angle),z=Math.sin(angle);
          const point=center.clone().addScaledVector(across,sign*width*c).addScaledVector(depth,width*1.1*z);
          if(ring>=8){
            const seam=hips.clone().addScaledVector(hipAxis,sign*Math.max(0,c)*(span+.044))
              .addScaledVector(hipBack,.099*z).addScaledVector(hipUp,-.025+.069*Math.min(0,c));
            point.lerp(seam,ring===8?.42:1);
          }
          positions.setXYZ((side*10+ring)*24+j,point.x,point.y,point.z);
        }
      }
    }
    const hipWidth=p[23].distanceTo(p[24])*.5+.044;
    const lowerQ=this.hips.quaternion,upperQ=this.torso.quaternion;
    for(let r=0;r<9;r++){
      const u=[.005,.085,.12,.26,.47,.70,.87,.96,1.02][r];
      const q=lowerQ.clone().slerp(upperQ,T.MathUtils.smoothstep(u,.06,.85));
      const width=[hipWidth,hipWidth*.96,hipWidth*.965,chest*.80,chest*.88,chest*.99,chest*1.02,chest*.88,.052][r];
      const depth=[.104,.103,.106,.108,.118,.129,.121,.102,.051][r];
      const center=hips.clone().lerp(shoulders,u);
      for(let j=0;j<24;j++){
        const a=j*Math.PI/12;
        const point=new T.Vector3(width*Math.cos(a),0,depth*Math.sin(a)).applyQuaternion(q).add(center);
        positions.setXYZ((20+r)*24+j,point.x,point.y,point.z);
      }
    }
    positions.needsUpdate=true;this.upperLegs.geometry.computeVertexNormals();
    const hand=p[15].clone().add(p[16]).multiplyScalar(.5);
    
    // Club display & realistic swing kinematics
    for(const object of [this.shaft,this.grip,this.clubhead])object.visible=!this.noPose;
    if(!this.noPose){
      let direction;
      if(club){
        const raw=this.reverseClubDepth?club.alternate_direction:club.direction;
        direction=new T.Vector3(raw[0],-raw[1],-raw[2]).normalize();
      } else {
        // Carry address shaft orientation with the lead arm. Wrist cock increases
        // continuously as the hands lift, on both sides of impact.
        const address=this.poseAt(row,0);
        const baseArm=address[15].clone().sub(address[11]).normalize();
        const arm=p[15].clone().sub(p[11]).normalize();
        const baseHand=address[15].clone().add(address[16]).multiplyScalar(.5);
        baseHand.add(this.stanceOffset);
        baseHand.y+=.09-Math.min(address[27].y,address[28].y);
        const rotation=new T.Quaternion().setFromUnitVectors(baseArm,arm);
        const addressDir=this.ballOrigin.clone().sub(baseHand).normalize();
        direction=addressDir.applyQuaternion(rotation);
        const hingeAxis=baseArm.clone().cross(arm);
        const turn=baseArm.angleTo(arm);
        if(hingeAxis.lengthSq()>.0001)
          direction.applyAxisAngle(hingeAxis.normalize(),Math.min(1.32,turn*.82));
      }
      const clubLength=row.club_display?.club_length_display_units||.94615;
      if(!club){
        const offset=this.ballOrigin.clone().sub(hand);
        const atContact=Math.exp(-Math.pow((t-this.impact)/.105,2));
        const towardBall=offset.clone().normalize();
        direction.lerp(towardBall,atContact).normalize();
        const reachError=offset.length()-(clubLength-.079);
        hand.addScaledVector(towardBall,atContact*T.MathUtils.clamp(reachError,-.075,.075));
      }
      if(row.club_motion){
        const motion=row.club_motion;
        const keys=(this.reverseClubDepth && motion.alternate) || motion.direction;
        const index=T.MathUtils.clamp(t*motion.fps,0,keys.length-1);
        const offsets=(this.reverseClubDepth && motion.alternate_hand_offset) || motion.hand_offset;
        if(offsets)hand.add(new T.Vector3().fromArray(offsets[Math.floor(index)])
          .lerp(new T.Vector3().fromArray(offsets[Math.ceil(index)]),index%1));
        direction.fromArray(keys[Math.floor(index)])
          .lerp(new T.Vector3().fromArray(keys[Math.ceil(index)]),index%1).normalize();
      }
      // Anchor the center of the striking face, which sits outboard of the hosel.
      // Let the filmed motion take over continuously as the hands leave address.
      const setup=1-T.MathUtils.smoothstep(t,this.takeaway,this.takeaway+.42);
      const contact=row.departure_interval?1-T.MathUtils.smoothstep(Math.abs(t-this.impact),0,.18):0;
      const weight=Math.max(setup,contact);
      if(weight>0){
        const faceTarget=this.ballOrigin.clone();
        faceTarget.z+=.028+.012*setup;
        const reach=clubLength-.079;
        let aimed=faceTarget.clone().sub(hand).normalize();
        let hosel;
        for(let i=0;i<3;i++){
          const localFace=this.facePlate.position.clone().applyQuaternion(this.squareIron(aimed));
          hosel=faceTarget.clone().sub(localFace);
          aimed=hosel.clone().sub(hand).normalize();
        }
        hand.lerp(hosel.addScaledVector(aimed,-reach),weight);
        let reference=direction;
        if(row.club_motion && (contact>setup || t>this.takeaway)){
          // Fade a correction relative to the takeaway direction. Recomputing the
          // shortest rotation from the moving shaft can flip axes near 180 degrees.
          const motion=row.club_motion;
          const keys=(this.reverseClubDepth && motion.alternate)||motion.direction;
          const anchorTime=contact>setup?this.impact:this.takeaway;
          const i=Math.min(keys.length-1,anchorTime*motion.fps);
          reference=new T.Vector3().fromArray(keys[Math.floor(i)])
            .lerp(new T.Vector3().fromArray(keys[Math.ceil(i)]),i%1).normalize();
        }
        const turn=new T.Quaternion().setFromUnitVectors(reference,aimed);
        direction.applyQuaternion(new T.Quaternion().slerp(turn,weight)).normalize();
      }
      // Place both palms along the same rigid handle, then terminate each
      // forearm at its palm. The club butt starts behind the lead hand.
      for(let k=0;k<2;k++){
        const palm=hand.clone().addScaledVector(direction,k?.031:-.031);
        this.hands[k].position.copy(palm);
        this.hands[k].scale.set(.84,1.23,.91);
        this.hands[k].quaternion.setFromUnitVectors(new T.Vector3(0,1,0),direction);
        p[15+k].copy(palm);
      }
      const butt=hand.clone().addScaledVector(direction,-.079);
      const tip=butt.clone().addScaledVector(direction,clubLength);
      this.segment(this.shaft,butt,tip);
      this.segment(this.grip,butt,butt.clone().addScaledVector(direction,.255));
      this.clubhead.position.copy(tip);
      this.clubhead.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),direction.clone().negate());
      const faceAngleDeg=row.club_fusion?.metrics?.dynamic_face_angle_deg||0;
      this.clubhead.rotateY(.25 + faceAngleDeg * Math.PI / 180);
      if(weight>0)this.clubhead.quaternion.slerp(this.squareIron(direction),weight);
    }

    for(const part of this.parts)this.segment(part.mesh,p[part.a],p[part.b]);
    this.airSeconds=Math.max(0,clock-this.launchTime);
    for(const f of this.flights)this.updateFlight(f,this.airSeconds);
    for(const key of ['Truth','Pred']){
      if(this['has'+key])continue;
      for(const prefix of ['ball','trace','ribbon','landRing','disc','outerRing','totalRing'])
        this[prefix+key].visible=false;
    }
    this.frameCamera();
    for(const f of this.flights){
      const pts=[],geom=this['traceGeometry'+f.key];
      if(this.airSeconds>0){
        for(const sample of f.trail){
          if(sample.time>=Math.min(this.airSeconds,f.hang+f.rollTime)-1e-9)break;
          pts.push(sample.point);
        }
        pts.push(this['ball'+f.key].position);
      }
      f.visibleTrail=pts;
      pts.forEach((point,j)=>geom.attributes.position.setXYZ(j,point.x,point.y,point.z));
      geom.attributes.position.needsUpdate=true;geom.setDrawRange(0,pts.length);
      this.updateRibbon(this['ribbonGeom'+f.key],pts);
    }
    const prediction=this.flights.find(f=>f.key==='Pred');
    this.travelYards=prediction?.yards??null;
    this.flightProgress=prediction?.q??0;
    this.drawMarkers();
    this.renderer.render(this.scene,this.camera);
  }
};
