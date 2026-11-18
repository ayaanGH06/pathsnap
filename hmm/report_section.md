# HMM decoder: design, tuning, and final evaluation

## Decoder design

Each hidden state is a directed road arc from the candidate set for one GPS observation. The saved HistGradientBoosting model supplies candidate-match probabilities as emission scores. The Viterbi decoder combines those scores across the trip and prefers transitions whose directed-network route distance is consistent with the GPS displacement and elapsed time. It penalizes reverse-twin turns and permits repeated arcs.

The frozen settings are in [`hmm_config.json`](hmm_config.json). Key values include emission weight 0.5, a maximum of 20 candidates per observation, transition scale `beta_m=5`, reverse-twin penalty 15, a 120-second chain restart, and a maximum plausible speed of 45 m/s. If no candidate is available, the decoder can carry the nearest decoded neighbour's arc for up to 30 seconds. Filled values use predictions and timestamps only; they are still counted in the full observation denominator and are correct only when they match the reference.

## Tuning protocol

Decoder settings were selected on 20 of the 80 training trips using out-of-fold scores. The tuning objective excluded trips 00000033 and 00000054 because their 50 m candidate coverage is unusually poor; the all-trip results were retained and reported as well. These are development-selection results, not an independent estimate of performance. Student-test and evaluator-test labels were not used to tune the decoder.

The final student-test evaluation used B's saved model artifact and the frozen decoder configuration. The evaluator-test was a separate frozen run: no fitting, tuning, or selection used its labels, and each reference route was opened only after prediction. The run evidence is saved in [`results/hmm/final_student_test_artifact.json`](../results/hmm/final_student_test_artifact.json) and [`results/hmm/evaluator_frozen_run_evidence.json`](../results/hmm/evaluator_frozen_run_evidence.json).

## Results

Pointwise accuracy is computed over every GPS observation. Missing or incorrect predictions count as wrong. The two partitions are separate runs and must not be averaged.

| Partition | Observations | Classifier alone | Classifier + HMM | HMM correct |
|---|---:|---:|---:|---:|
| Student-test, 10 trips | 19,160 | 85.80% | 93.52% | 17,919 / 19,160 |
| Evaluator-test, 10 trips | 19,674 | 79.93% | 94.72% | 18,636 / 19,674 |

On student-test, mean per-trip accuracy increased from 87.41% to 93.94%. Direction flips between consecutive predictions fell from 1,308 to 9, and the predicted-to-reference segment-count ratio fell from 1.85 to 1.04. The HMM improved pooled accuracy, but it did not help every trip: trip 00000007 scored 96.01% with the classifier alone and 89.07% with the decoder. Per-trip results and other diagnostics are in the saved evidence and the integrated report section [`report/section7.md`](../report/section7.md).

On evaluator-test, 21 observations had no candidate within 50 m; all were on trip 00000015. The frozen carry rule filled all 21 from neighbouring decoded predictions, and 12 of those fills matched the reference. All remain in the denominator. The HMM reached 94.72% overall, with per-trip accuracy from 88.53% to 99.71%; this is one frozen run, not a guarantee for every trip.

## Limitations

The decoder can only choose among roads supplied by candidate generation. It cannot recover a correct road missing from the 50 m candidate pool. Direction remains a weakness: geometrically overlapping forward and reverse arcs can confuse the classifier and decoder, as trip 00000007 shows. Macro set overlap and segment counts describe route structure but are not full sequence accuracy; the headline measure here is pointwise accuracy.
