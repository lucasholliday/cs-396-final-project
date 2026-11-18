"""Verify original source hashes and scientific-output consistency without decoding video."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
inventory=pd.read_csv(ROOT/'data/manifests/file_inventory.csv')
for row in inventory.itertuples():
    h=hashlib.sha256()
    with (ROOT/row.path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=row.sha256:raise AssertionError(f'Source modified: {row.path}')
manifest=pd.read_csv(ROOT/'data/manifests/shot_manifest.csv')
assert manifest.shot_id.is_unique
review=pd.read_csv(ROOT/'annotations/shot_review.csv')
assert set(review.shot_id)==set(manifest.shot_id)
pose=pd.read_csv(ROOT/'data/processed/pose_features.csv')
assert set(pose.shot_id)==set(manifest.shot_id)
assert manifest.loc[manifest.training_eligible,'trackman_status'].eq('registered').all()
for experiment in ['baseline_v1','nested_carry_v1']:
    p=pd.read_csv(ROOT/'experiments'/experiment/'predictions.csv')
    keys=['cohort','scheme','features','shot_id']
    if experiment=='baseline_v1':keys+=['target','model']
    assert not p.duplicated(keys).any()
    assert set(p.shot_id)<=set(manifest.loc[manifest.training_eligible,'shot_id'])
summary=dict(original_files_verified=len(inventory),source_bytes_verified=int(inventory.bytes.sum()),
    reviewed_shots=len(review),pose_shots=len(pose),eligible_outcomes=int(manifest.training_eligible.sum()),
    status='passed',checks='SHA256 originals; unique IDs; review/pose coverage; no unlabeled predictions; OOF uniqueness')
(ROOT/'reports/checkpoint_verification.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
