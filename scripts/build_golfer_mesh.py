"""Bake the CC0 MakeHuman mesh into an offline range asset. Inputs: base.obj, default_weights.mhw and an adult body target."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

vertices, groups = [], defaultdict(list)
for line in Path(sys.argv[1]).read_text().splitlines():
    words = line.split()
    if not words:
        continue
    if words[0] == 'v':
        vertices.append([float(x)*.1 for x in words[1:4]])
    elif words[0] == 'g':
        group = words[1]
    elif words[0] == 'f':
        groups[group].append([int(x.split('/')[0])-1 for x in words[1:]])

v = np.array(vertices)
def joint(name):
    return v[np.unique(groups['joint-'+name])].mean(axis=0)

faces = defaultdict(list)
for group in ['helper-tights','body','helper-l-eye','helper-r-eye']:
    for polygon in groups[group]:
        x,y,z = v[polygon].mean(axis=0)
        arm = abs(x)>.26 and y>.10
        if group=='helper-tights':
            if y<-.74 or (arm and abs(x)>.41): continue
            material = 'shirt' if y>.18 or arm else 'pants'
            if .15<y<=.18 and not arm: material='belt'
        elif group=='body':
            if y<.55 and not (arm and abs(x)>.39 and abs(x)<.443 and z<.194): continue
            material='skin'
            if y>.60 and .612<y<.641 and .115<z and abs(x)<.028: material='lip'
        else:
            material='eye'
            if z>joint(group.replace('helper-',''))[2]+.010: material='iris'
        for i in range(1,len(polygon)-1):
            faces[material].extend([polygon[0],polygon[i],polygon[i+1]])
# Select garment boundaries on the canonical topology, then morph both the
# cloth and its joint markers into the same adult reference proportions.
target=np.loadtxt(sys.argv[3])
v[target[:,0].astype(int)]+=target[:,1:4]*.1
bones = []
def bone(a,b):
    bones.append([a,b])
s = (joint('l-shoulder')+joint('r-shoulder'))/2
hip = (joint('l-upper-leg')+joint('r-upper-leg'))/2
waist = hip*.75+s*.25
bone(hip,waist)
bone(waist,s)
bone(s,joint('head'))
bone(joint('head'),joint('head')+[0,.14,0])
for side in ['l-','r-']:
    for a,b in [('shoulder','elbow'),('elbow','hand'),('upper-leg','knee'),('knee','ankle')]:
        bone(joint(side+a),joint(side+b))
for side in ['l-','r-']:
    bone(s,joint(side+'shoulder'))
bones = np.array(bones)
ids=np.unique([i for f in faces.values() for i in f])
lookup={k:i for i,k in enumerate(ids)}
weights,indices=[],[]
# Collapse the artist-authored MakeHuman weights onto the display skeleton.
binding=np.zeros((len(v),len(bones)))
for name,entries in json.loads(Path(sys.argv[2]).read_text())['weights'].items():
    side=int(name.endswith('.R'))
    if name.startswith('upperarm'): index=4+4*side
    elif name.startswith(('lowerarm','wrist','finger','metacarpal')): index=5+4*side
    elif name.startswith('upperleg'): index=6+4*side
    elif name.startswith(('lowerleg','foot','toe')): index=7+4*side
    elif name.startswith(('clavicle','shoulder')): index=12+side
    elif name.startswith(('root','pelvis','spine04','spine05')): index=0
    elif name.startswith(('breast','spine')): index=1
    elif name.startswith('neck'): index=2
    else: index=3
    for vertex,weight in entries:
        binding[vertex,index]+=weight
for vertex in ids:
    sel=np.argsort(binding[vertex])[-4:][::-1]
    w=binding[vertex,sel]
    assert w.sum()>0
    weights.extend(np.round(w/w.sum(),6));indices.extend(sel.tolist())
# Shape cloth around horizontal sections. A continuous seat panel spans the
# pelvis, while the legs taper gradually from knee to an open trouser hem.
tailored=v[ids].copy()
pants_ids=set(faces['pants'])
shirt_ids=set(faces['shirt'])|set(faces['belt'])
def smooth(lo,hi,x):
    t=np.clip((x-lo)/(hi-lo),0,1)
    return t*t*(3-2*t)

for i,vertex in enumerate(ids):
    x,y,z=tailored[i]
    if vertex in pants_ids:
        side='l-' if x>=0 else 'r-'
        leg=[joint(side+k) for k in ['ankle','knee','upper-leg']]
        cx,cz=[np.interp(y,[q[1] for q in leg],[q[axis] for q in leg]) for axis in [0,2]]
        # Follow the trouser crease through the instep, knee and thigh.
        angle=np.arctan2(z-cz,x-cx)
        levels=[-.77,-.69,-.51,-.31,-.12,.04]
        width=np.interp(y,levels,[.066,.065,.068,.074,.083,.093])
        depth=np.interp(y,levels,[.073,.071,.075,.084,.097,.109])
        fold=.0025*np.cos(6*angle+.6*y)
        leg_x=cx+(width+fold)*np.cos(angle)
        leg_z=cz+(depth+fold)*np.sin(angle)
        separate=1-smooth(-.11,.055,y)
        tailored[i,0]=x+(leg_x-x)*separate
        tailored[i,2]=z+(leg_z-z)*separate
        # Curved back panel bridges the seat's center groove.
        if z<.015:
            back=-.104*np.sqrt(max(0,1-(x/.196)**2))
            coverage=smooth(-.055,.075,y)*(1-smooth(.17,.23,y))
            tailored[i,2]+=coverage*min(0,back-tailored[i,2])
    if vertex in shirt_ids:
        # Ease and length in the lower polo let the hem hang from the chest.
        lower=1-smooth(.30,.52,y)
        if abs(x)<.25:
            angle=np.arctan2((z-.027)/.115,x/.17)
            tailored[i,0]+=lower*.024*np.cos(angle)
            tailored[i,2]+=lower*.023*np.sin(angle)
            tailored[i,1]-=.05*lower
        elif abs(x)>.25:
            side='l-' if x>0 else 'r-'
            arm=np.array([joint(side+k) for k in ['shoulder','elbow','hand']])
            center=np.array([np.interp(abs(x),abs(arm[:,0]),arm[:,j]) for j in [1,2]])
            radial=np.array([y,z])-center
            ease=.009*smooth(.25,.32,abs(x))*(1-smooth(.44,.49,abs(x)))
            tailored[i,1:]+=ease*radial/max(.005,np.linalg.norm(radial))

# A separate hem overlays the waistband, so the bottom of the shirt can
# hang freely without dragging the top row of trouser vertices down.
for vertex in shirt_ids & pants_ids:
    i=lookup[vertex]
    new=len(tailored)
    tailored=np.vstack((tailored,v[vertex]))
    weights.extend(weights[4*i:4*i+4]);indices.extend(indices[4*i:4*i+4])
    # Store synthetic IDs only in the trouser triangles.
    synthetic=len(v)+vertex
    lookup[synthetic]=new
    faces['pants']=[synthetic if n==vertex else n for n in faces['pants']]
faces['shirt'].extend(faces.pop('belt'))
asset=dict(bones=bones.round(5).tolist(),positions=tailored.round(5).flatten().tolist(),weights=weights,
           joints=indices,faces={m:[lookup[i] for i in f] for m,f in faces.items()})
dest=Path(__file__).resolve().parents[1]/'vendor/golfer/human.json'
dest.write_text(json.dumps(asset,separators=(',',':')))
