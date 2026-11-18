# Selected model handoff

`pathsnap_selected_model.joblib` is the selected HistGradientBoosting sklearn
Pipeline. It includes the median imputer and classifier. It was fit on the full
train candidate table after configuration selection; the student-test partition
was not used for hyperparameter tuning.

## Candidate scoring contract

1. A's feature pipeline must provide a pandas DataFrame with exactly these
   columns in this order (the order is also in `model_manifest.json`):
   `dist_to_arc`, `proj_frac`, `arc_len`, `heading_diff`, `speed`, `dt_prev`,
   `dist_rank`, `dist_minus_min`, `n_cand`, `conn_prev`.
2. Call `predict_proba(features)` for each candidate row.
3. Select the probability column whose class label is `1`. This is the score
   that the candidate matches the reference road. Group rows by observation and
   sort descending by this score for classifier-only Top-1.
4. For C's decoder, pass per-observation candidate probabilities in candidate
   row order as emission evidence. The probabilities are discriminative scores;
   they are not claimed to be a generative GPS likelihood.

Example:

```python
import joblib
import pandas as pd

model = joblib.load("models/artifacts/pathsnap_selected_model.joblib")
features = pd.DataFrame(candidate_feature_rows)
positive_col = list(model.classes_).index(1)
scores = model.predict_proba(features)[:, positive_col]
```

The artifact was serialized with the scikit-learn version recorded in
`model_manifest.json`. Load it with a compatible scikit-learn version; sklearn
does not guarantee pickle compatibility across versions. The preprocessing is
inside the artifact, so do not scale the features separately.
