"""Source-bound visual club anchors for a display pilot; never model inputs."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def complete(p,review):
    world=np.asarray(p['estimated_world_landmarks'],float);lm=np.asarray(p['landmarks'],float);size=np.array(review['image_size']);n=len(world)
    # Estimate one camera orientation/scale for the clip, with per-frame hip translation removed.
    xx=[];yy=[]
    for i in range(n):
        wc=world[i,[23,24]].mean(0);ic=lm[i,[23,24],:2].mean(0)*size
        for j in [11,12,23,24,25,26,27,28]:
            if lm[i,j,3]>.6 and np.isfinite(world[i,j]).all():xx.append(world[i,j]-wc);yy.append(lm[i,j,:2]*size-ic)
    if len(xx)<20:return None
    a=np.linalg.lstsq(np.array(xx),np.array(yy),rcond=None)[0].T
    u,s,vh=np.linalg.svd(a,full_matrices=False)
    # A projected shaft cannot exceed its assumed display length. This lower bound
    # on scale prevents clipping image evidence to an incompatible body-only fit.
    longest=max(np.linalg.norm(np.array(r['head'])-r['grip']) for r in review['points'])
    scale=max(float(np.mean(s)),longest/.85)
    projection=(u@vh)*scale;inverse=np.linalg.pinv(projection)
    normal=np.cross(projection[0],projection[1]);normal/=np.linalg.norm(normal)
    anchors={r['frame']:r for r in review['points']};keys=sorted(anchors);records=[];previous=None
    for i in range(n):
        ref=anchors.get(i);kind='approximate_visual_anchor'
        if ref is None:
            before=[j for j in keys if j<i];after=[j for j in keys if j>i]
            if before and after:
                lo,hi=before[-1],after[0];setup=[lo,hi]==review.get('stationary_setup_interval')
                if hi-lo<=review['maximum_interpolation_gap_frames'] or setup:
                    f=(i-lo)/(hi-lo);ref={k:(np.array(anchors[lo][k])*(1-f)+np.array(anchors[hi][k])*f).tolist() for k in ['grip','head']};kind='display_interpolation'
        if ref is None or i in review.get('occluded_frames',[]):records.append(None);previous=None;continue
        grip=np.array(ref['grip']);head=np.array(ref['head']);plane=inverse@(head-grip)
        # Both depth signs project to the same image. A fixed length is an avatar assumption.
        length=.85;plane_len=np.linalg.norm(plane)
        if plane_len>length:plane*=length/plane_len
        depth=np.sqrt(max(0,length**2-np.dot(plane,plane)))
        choices=[plane+normal*depth,plane-normal*depth]
        direction=min(choices,key=lambda x:np.linalg.norm(x-previous)) if previous is not None else max(choices,key=lambda x:x[1])
        previous=direction;raw_hand=world[i,[15,16]].mean(0);image_hand=lm[i,[15,16],:2].mean(0)*size
        shift=inverse@(grip-image_hand)
        records.append(dict(direction=(direction/length).tolist(),alternate_direction=((2*plane-direction)/length).tolist(),hand_shift=shift.tolist(),grip=grip.tolist(),head=head.tolist(),status=kind))
    return dict(frames=records,projection=projection.tolist(),body_fit_scale=float(np.mean(s)),display_scale=scale,club_length_display_units=.85,semantics='Assisted visual pilot. Source-image anchors with short display interpolation. Orthographic display scale is constrained by the longest visible shaft; depth sign and fixed club length are assumptions, not calibrated 3D. Second reviewer pending.')

if __name__=='__main__':
    out=ROOT/'data/processed/club_display';out.mkdir(exist_ok=True)
    for path in (ROOT/'annotations').glob('club_keypoints_*.json'):
        review=json.loads(path.read_text());sid=review['shot_id'];video=ROOT/f'outputs/pose/{sid}_input.mp4'
        assert hashlib.sha256(video.read_bytes()).hexdigest()==review['input_sha256']
        p=json.loads((ROOT/f'data/processed/pose/{sid}.json').read_text());result=complete(p,review)
        (out/f'{sid}.json').write_text(json.dumps(result));print(sid,sum(r is not None for r in result['frames']),len(result['frames']))
