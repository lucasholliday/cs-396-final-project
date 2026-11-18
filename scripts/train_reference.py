"""Save fixed ridge reference fits; evaluation remains the separate out-of-fold experiment."""
import hashlib
import json
from pathlib import Path
import joblib
from benchmark import load_data,model

ROOT=Path(__file__).resolve().parents[1]
cfg,d=load_data();spec=json.loads((ROOT/'experiments/baseline_v1/config.json').read_text())
out=ROOT/'models/reference_ridge';out.mkdir(parents=True,exist_ok=True)
manifest=[]
for name in ['camera','watch','all']:
    cols=spec['feature_sets'][name]
    subset=d.dropna(subset=spec['feature_sets']['club']) if name=='all' else d
    for target in cfg['targets']:
        estimator=model('ridge10',cfg['seed']);estimator.fit(subset[cols],subset[target])
        slug=target.split(' (')[0].lower().replace(' ','_')
        path=out/f'{name}_{slug}.joblib'
        joblib.dump(dict(model=estimator,features=cols,target=target,semantics=cfg['description']),path)
        manifest.append(dict(path=str(path.relative_to(ROOT)),features=name,target=target,n=len(subset),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),status='full_data_reference_fit_not_independent_validation'))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Saved',len(manifest),'reference fits; use OOF reports for performance, not training error.')
