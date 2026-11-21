// Sample actual displayed shaft, grip and collision geometry, including subframes.
const fs=require('fs'),path=require('path'),vm=require('vm');
const root=path.resolve(__dirname,'..'),T=global.THREE=require(path.join(root,'vendor/three/three.min.js'));
global.window={THREE:T,devicePixelRatio:1};
T.WebGLRenderer=class {constructor(){this.shadowMap={}}setPixelRatio(){}setSize(){}render(){}};
vm.runInThisContext(fs.readFileSync(path.join(root,'scripts/range_scene.js'),'utf8'));
const data=JSON.parse(fs.readFileSync(path.join(root,'outputs/research_viewer/index.html'),'utf8').match(/const data=(.*?);const \$/)[1]);
const dir=path.join(root,'outputs/club_motion',process.argv.includes('--check')?'after':'before');
fs.mkdirSync(dir,{recursive:true});
const rig=new window.SwingRange({clientWidth:720,clientHeight:600,style:{},addEventListener(){}});
for(const row of data){
  if(!process.argv.includes('--check'))delete row.club_motion;
  for(const alternate of (row.club_display?[false,true]:[false])){
  rig.reverseClubDepth=alternate;
  const frames=[];
  for(let i=0;i<=4*(row.world.length-1);i++){
    const t=i/(4*row.fps);rig.update(row,t,0);
    const hand=rig.hands[0].position.clone().lerp(rig.hands[1].position,.5);
    const direction=rig.clubhead.position.clone().sub(hand).normalize();
    frames.push([hand.toArray(),direction.toArray(),rig.torso.position.toArray(),rig.torso.quaternion.toArray(),
      rig.torso.scale.toArray(),rig.head.position.toArray(),rig.hips.position.toArray()]);
  }
  fs.writeFileSync(path.join(dir,row.id+(alternate?'_alternate':'')+'.json'),JSON.stringify({frames,start:row.start,fps:row.fps*4}));
}
}
