# Section 7: System Integration and Evaluation

## Implemented system functionality

The Person D contribution provides a FastAPI foundation with health checking,
partition-aware trip listing and prediction request validation. The API loads
student-test or training trips through the existing shared loader with
`with_route=False`, so reference route labels are not loaded or exposed.

The frontend provides trip selection, prediction controls, loading, success
and error states, predicted-arc rendering, GPS observations and an optional
server metrics display. It uses no synthetic routes or invented performance
numbers.

The system supports two-laptop deployment through an environment-configurable
backend URL and CORS origin list.

## Prediction status

The real feature pipeline, KNN model, and HMM decoder are not yet committed in
the API integration path. `/predict` therefore returns HTTP 503 until the
shared team contracts are delivered. This is deliberate: the API must not
fabricate predictions or accuracy values.

## KNN experimental status

The assigned model key is `knn_distance_weighted`. The real candidate tables
`data/processed/candidates_train.parquet` and
`data/processed/candidates_student_test.parquet` are not currently available,
so no real KNN metrics or experimental findings are claimed. The experiment
will run only after Person A hands off both tables.
