"""Score displayed endpoints against means fitted on the same outer training golfers."""
import pandas as pd
import numpy as np

COORDS = ["Carry flat (yd)", "Carry side (yd; R+)", "Total (yd)", "Total side (yd; R+)"]


def score_cohorts(rows, predictions, labels):
    actual = labels.rename(columns={"Carry side (ft; R+)": COORDS[1], "Total side (ft; R+)": COORDS[3]}).copy()
    actual = actual.dropna(subset=COORDS)
    actual[COORDS[1::2]] /= 3
    means = pd.DataFrame(index=actual.index, columns=COORDS, dtype=float)
    for golfer in actual.Golfer.unique():
        mask = actual.Golfer == golfer
        means.loc[mask] = actual.loc[~mask, COORDS].mean().to_numpy()
    frame = pd.DataFrame(rows).set_index("id")
    primary = frame.used_primary_model
    shown = frame.best_prediction.notna()
    results = []
    for cohort, mask in [("Primary geometry",primary),("Accepted primary",primary & shown),("All displayed",shown)]:
        ids = frame.index[mask]
        truth = actual.loc[ids,COORDS].to_numpy()
        def errors(values):
            delta = values.loc[ids,COORDS].to_numpy()-truth
            return [abs(delta[:,0]).mean(),abs(delta[:,2]).mean(),
                    np.linalg.norm(delta[:,:2],axis=1).mean(),np.linalg.norm(delta[:,2:],axis=1).mean()]
        for metric,model,baseline in zip(["Carry distance","Total distance","2D landing position","2D final position"],errors(predictions),errors(means)):
            results.append(dict(cohort=cohort,shots=len(ids),metric=metric,model=model,baseline=baseline,reduction=100*(1-model/baseline)))
    return pd.DataFrame(results)
