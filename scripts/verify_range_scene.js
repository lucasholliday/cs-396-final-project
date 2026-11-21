/* Pure scene-graph/math tests. No browser, UI automation, WebGL or screenshot. */
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(__dirname,'..');global.THREE=require(path.join(root,'vendor/three/three.min.js'));global.window={devicePixelRatio:1,THREE};
THREE.WebGLRenderer=class GeometryOnlyRenderer {constructor(){this.shadowMap={}}setPixelRatio(){}setSize(){}render(scene,camera){scene.updateMatrixWorld(true);camera.updateMatrixWorld(true)}};
vm.runInThisContext(fs.readFileSync(path.join(root,'scripts/range_scene.js'),'utf8'));
const html=fs.readFileSync(path.join(root,'outputs/research_viewer/index.html'),'utf8'),data=JSON.parse(html.match(/const data=(.*?);const \$/)[1]);
const canvas={clientWidth:720,clientHeight:600,style:{},addEventListener(){}};const rig=new window.SwingRange(canvas);assert(rig.ready);let frames=0,maxHandGap=0,maxPalmOffset=0,maxWristGap=0,maxHoselGap=0;
for(const row of data){assert(row.avatar_motion.frames.length===105);for(let i=0;i<105;i++){rig.update(row,i/row.fps,25);assert(rig.parts.every(p=>p.mesh.visible && p.mesh.matrixWorld.elements.every(Number.isFinite)));assert(rig.shoes.every(s=>s.position.y>=.085 && new THREE.Vector3(0,0,1).applyQuaternion(s.quaternion).x>0));assert(rig.shaft.visible&&rig.clubhead.visible);if(Number.isFinite(row.best_prediction)){assert(rig.travelYards>=0);assert(rig.travelYards<=Math.max(rig.totalPred,rig.carryPred)+1)}else assert(rig.travelYards===null);maxHandGap=Math.max(maxHandGap,rig.hands[0].position.distanceTo(rig.hands[1].position));const localAxis=new THREE.Vector3(0,1,0);
for(let side=0;side<2;side++){
  const palm=rig.hands[side].position;
  const onGrip=rig.grip.worldToLocal(palm.clone());
  assert(Math.abs(onGrip.y)<=.5);
  maxPalmOffset=Math.max(maxPalmOffset,Math.hypot(onGrip.x,onGrip.z));
  const forearm=rig.parts.find(part=>part.a===13+side&&part.b===15+side);
  const wrist=forearm.mesh.localToWorld(localAxis.clone().multiplyScalar(.5));
  maxWristGap=Math.max(maxWristGap,wrist.distanceTo(palm));
}
maxHoselGap=Math.max(maxHoselGap,rig.shaft.localToWorld(localAxis.clone().multiplyScalar(.5)).distanceTo(rig.clubhead.position));
frames++}}
assert(Math.abs(maxHandGap-.062)<1e-6);
assert(Math.max(maxPalmOffset,maxWristGap,maxHoselGap)<1e-6);
const pilot=data.find(r=>r.club_display);rig.update(pilot,0,25);assert(rig.travelYards===0);rig.update(pilot,(pilot.world.length-1)/pilot.fps,25);assert(Math.abs(rig.travelYards-rig.totalPred)<1e-6);rig.update(pilot,0,25);assert(rig.travelYards===0);
// Check all initial views and both follow targets at narrow and wide aspect ratios.
function inFrame(point){
  const v=point.clone().project(rig.camera);
  assert(Math.abs(v.x)<1 && Math.abs(v.y)<1 && Math.abs(v.z)<1, JSON.stringify(v));
}
for(const width of [320,900]){
  canvas.clientWidth=width;
  for(const row of data){
    rig.update(row,0,0,true);
    for(const time of [0,rig.takeaway*.5,rig.takeaway]){
      rig.update(row,time,0,true);
      const face=rig.facePlate.getWorldPosition(new THREE.Vector3()).sub(rig.ballOrigin);
      assert(Math.abs(face.x)<.002 && Math.abs(face.y)<.002 && Math.abs(face.z-.04)<.002);
    }
    rig.update(row,0,0,true);
    for(const shoe of rig.shoes){
      const toeEdge=new THREE.Box3().setFromObject(shoe,true).max.x;
      assert(rig.ballOrigin.x-toeEdge>.44);
    }
    for(const mode of ['side_wide','range','landing','follow']){
      rig.cameraMode=mode;rig.update(row,0,0,true);
      for(const mesh of [rig.head,...rig.shoes,rig.clubhead])inFrame(mesh.position);
      inFrame(rig.ballOrigin);
    }
    for(let seconds=0;seconds<=rig.endTime+.1;seconds+=.14){
      rig.update(row,seconds,0,true);
      for(const f of rig.flights)inFrame(rig['ball'+f.key].position);
    }
    if(rig.flights.length){
      const ctx={setTransform(){},clearRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},
        fillRect(){},fillText(){},measureText:text=>({width:text.length*6.5})};
      rig.markerOverlay={getContext:()=>ctx};
      rig.update(row,rig.launchTime+rig.firstLanding+.0001,0,true);
      const layout=rig.markerLayout;
      assert(layout.labels.length===2*rig.flights.length);
      rig.update(row,rig.endTime,0,true);
      assert.strictEqual(rig.markerLayout,layout);
      for(const f of rig.flights){
        for(const prefix of ['outerRing','totalRing']){
          const ring=rig[prefix+f.key],radius=ring.geometry.parameters.outerRadius;
          for(let i=0;i<64;i++){
            const a=i*Math.PI/32;
            inFrame(ring.position.clone().add(new THREE.Vector3(radius*Math.cos(a),0,radius*Math.sin(a))));
          }
        }
      }
      const labels=layout.labels;
      for(let i=0;i<labels.length;i++){
        const a=labels[i];
        for(const b of labels.slice(i+1))assert(a.x+a.width<=b.x||b.x+b.width<=a.x||a.y+19<=b.y||b.y+19<=a.y);
      }
      rig.markerOverlay=null;
    }
    rig.cameraMode='side_wide';
    rig.update(row,0,0,true);
    const fixedCamera=rig.camera.position.clone();
    for(let seconds=0;seconds<=rig.endTime+.1;seconds+=.14){
      rig.update(row,seconds,0,true);
      assert(fixedCamera.distanceTo(rig.camera.position)<1e-6);
      inFrame(rig.head.position);
      for(const f of rig.flights)inFrame(rig['ball'+f.key].position);
    }
    rig.cameraMode='range';rig.update(row,0,0,true);
    for(const f of rig.flights){
      for(let i=0;i<=100;i++)inFrame(rig.flightPoint(f,i/100));
    }
    for(let t=0;t<rig.poseDuration;t+=.16){
      rig.update(row,t,0,false);
      const club=rig.clubhead.position.clone(),head=rig.head.position.clone();
      rig.update(row,t,0,true);
      assert(club.distanceTo(rig.clubhead.position)<1e-6 && head.distanceTo(rig.head.position)<1e-6);
    }
    for(const f of rig.flights){
      const touchdown=rig.launchTime+f.hang;
      rig.update(row,touchdown-.01,0,true);assert(f.q<1 && !rig['landRing'+f.key].visible);
      rig.update(row,touchdown,0,true);assert(rig['landRing'+f.key].visible);
      assert(Math.abs(rig['ball'+f.key].position.z-rig.ballOrigin.z+f.dist)<1e-6);
      rig.update(row,touchdown+f.hopTime/2,0,true);
      const ball=rig['ball'+f.key];
      assert(Math.abs(ball.position.y-rig.ballGroundY(ball.position.z)-f.hopHeight)<1e-6);
      rig.update(row,rig.endTime,0,true);assert(f.r===1);
      const line=rig['traceGeometry'+f.key],ribbon=rig['ribbonGeom'+f.key];
      for(const elapsed of [f.rollTime,f.hopTime*.5,f.hopTime+f.groundTime*.5,0]){
        rig.update(row,touchdown+elapsed,0,true);
        const count=line.drawRange.count;
        assert(count<=line.attributes.position.count);
        assert(ribbon.drawRange.count===(count-1)*6);
        const tip=new THREE.Vector3().fromBufferAttribute(line.attributes.position,count-1);
        assert(tip.distanceTo(ball.position)<.0001);
        if(elapsed>1e-6)assert(count>rig.trailSteps.air+1);
      }
      rig.update(row,0,0,true);
      assert(line.drawRange.count===0 && ribbon.drawRange.count===0);
      rig.update(row,rig.endTime,0,true);
      const side=f.key==='Pred'?row.best_total_side_ft:row.total_side_ft;
      const total=f.key==='Pred'?row.best_total_yd:row.total_actual;
      const expected=new THREE.Vector3(rig.ballOrigin.x+side*.3048,0,rig.ballOrigin.z-total*.9144);
      assert(Math.hypot(expected.x-ball.position.x,expected.z-ball.position.z)<1e-6);
      assert(Math.abs(f.yards-rig['total'+f.key])<1e-6);
    }
    rig.update(row,rig.poseDuration,0,true);const finish=rig.clubhead.position.clone();
    rig.update(row,rig.endTime,0,true);assert(finish.distanceTo(rig.clubhead.position)<1e-6);
  }
}
canvas.clientWidth=720;rig.cameraMode='range';
const out=path.join(root,'outputs/range_scene_preview');fs.mkdirSync(out,{recursive:true});
function snapshot(row,t,view,name){rig.view=view;rig.update(row,t,view==='range'?30:25);let polygons=[],lines=[];const camera=rig.camera,T=THREE,sun=new T.Vector3(-8,15,5).normalize();rig.scene.traverse(obj=>{if(!obj.visible)return;if(obj.isMesh&&!obj.material.wireframe){const g=obj.geometry,attr=g.attributes.position,index=g.index,material=Array.isArray(obj.material)?obj.material[0]:obj.material;for(let k=g.drawRange.start;k<Math.min(index?index.count:attr.count,g.drawRange.start+g.drawRange.count);k+=3){let verts=[];for(let j=0;j<3;j++){const ix=index?index.getX(k+j):k+j;verts.push(obj.getVertexPosition(ix,new T.Vector3()).applyMatrix4(obj.matrixWorld))}const norm=verts[1].clone().sub(verts[0]).cross(verts[2].clone().sub(verts[0])).normalize();if(material.side!==T.DoubleSide&&norm.dot(camera.position.clone().sub(verts[0]))<=0)continue;let cameraVerts=verts.map(v=>v.clone().applyMatrix4(camera.matrixWorldInverse));const clipped=[];for(let j=0;j<cameraVerts.length;j++){const a=cameraVerts[j],b=cameraVerts[(j+1)%cameraVerts.length],ain=a.z<-.05,bin=b.z<-.05;if(ain)clipped.push(a);if(ain!==bin){const u=(-.05-a.z)/(b.z-a.z);clipped.push(a.clone().lerp(b,u))}}if(clipped.length<3)continue;const depth=clipped.reduce((s,v)=>s+v.z,0)/clipped.length;const xy=clipped.map(v=>v.applyMatrix4(camera.projectionMatrix)).map(v=>[v.x,v.y]);const shade=.64+.36*Math.max(0,norm.dot(sun));const color=material.color.clone().multiplyScalar(shade).convertLinearToSRGB();polygons.push({xy,depth,color:[color.r,color.g,color.b]})}}else if(obj.isLine){const p=obj.geometry.attributes.position,count=Math.min(p.count,obj.geometry.drawRange.count);let xy=[];for(let j=0;j<count;j++){const v=new T.Vector3().fromBufferAttribute(p,j).applyMatrix4(obj.matrixWorld).project(camera);if(v.z>=-1&&v.z<=1)xy.push([v.x,v.y])}if(xy.length>1){const c=obj.material.color.clone().convertLinearToSRGB();lines.push({xy,color:[c.r,c.g,c.b]})}}});polygons.sort((a,b)=>a.depth-b.depth);fs.writeFileSync(path.join(out,name+'.json'),JSON.stringify({polygons,lines,view,shot:row.id,time:t,semantics:'Software projection of the actual scene graph for geometry inspection; not a browser screenshot or WebGL visual QA.'}));}
snapshot(data.find(r=>r.id==='sid_1_01'),.2,'golfer','address');snapshot(data.find(r=>r.id==='sid_1_01'),1.65,'golfer','swing');snapshot(data.find(r=>r.id==='sid_1_01'),2.75,'range','flight');
rig.cameraMode='side_wide';snapshot(data.find(r=>r.id==='lucas_1_09'),1.4,'range','overview');
const report={shots:data.length,frames_checked:frames,limbs_always_visible:true,finite_transforms:true,maximum_grip_hand_gap:maxHandGap,maximum_palm_axis_offset:maxPalmOffset,maximum_forearm_palm_gap:maxWristGap,maximum_shaft_hosel_gap:maxHoselGap,browser_or_webgl_test:false};fs.writeFileSync(path.join(root,'reports/range_scene_geometry_verification.json'),JSON.stringify(report,null,2));console.log(report);
