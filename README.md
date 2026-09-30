# cnn-lstm-imu-motion-recognition

Hybrid CNN–LSTM framework for classifying upper-limb rehabilitation movements from wrist-worn IMU data, with transfer-learning based per-session calibration.

This repository contains the implementation accompanying:

> S. H. Eshetu, S. T. Araya, A. B. Balcha, A. A. Damtie, A. M. Endale and Y. Benachour, "Hybrid CNN–LSTM Framework for Upper-Limb Motion Recognition Using Wearable IMU Sensors," Division of Engineering Technology & Science, Higher Colleges of Technology, Dubai, UAE.

**Please read [What the evaluation does and does not show](#what-the-evaluation-does-and-does-not-show) before citing any number from this repository.** The notebook's framing differs from the paper's abstract in one important respect, and the notebook is the version to rely on.

---

## What this does

Three upper-limb rehabilitation exercises are classified from a single wrist-mounted inertial sensor:

| Label | Movement |
|---|---|
| `el-exfl` | Elbow flexion / extension |
| `sh-exfl` | Shoulder flexion / extension |
| `wr-prsu` | Wrist pronation / supination |

A 1D-CNN extracts local spatial structure across sensor channels; an LSTM models the temporal evolution of the movement. A short supervised calibration then adapts the temporal layers to the deployment conditions while the convolutional layers stay frozen.

### Results

| Evaluation | Accuracy |
|---|---|
| Internal validation (random split of the training file) | 99.93% |
| External sessions, no calibration | **81.56%** (5-run mean 80.53% ± 2.27%) |
| External sessions, after ~60 s calibration, scored on a held-out half | **99.28%** |
| Best traditional baseline (Linear SVM), same protocol as the uncalibrated model | 74.26% |
| Logistic Regression / Random Forest | 68.66% / 59.68% |

The two numbers that matter are 81.56% and 99.28%. The first is what an uncalibrated model achieves on recording sessions it has not seen; the second is what the same model achieves after roughly a minute of labelled calibration data. The gap between them is the result — not the 99.93%.

---

## What the evaluation does and does not show

**The split is session-held-out, not subject-independent.** All three participants (`u01`, `u04`, `u05`) appear in the training data. What is held out is *recording sessions* — temporally distinct sittings excluded from training. The 81.56% figure therefore measures **session-to-session drift and concept drift**: the sensor being re-donned, and movement speed and amplitude varying between sittings. It says nothing about how the model would perform on a person it has never recorded.

With only three participants a leave-one-subject-out protocol was not feasible. That is the single most important methodological limitation of the study.

**How this maps to the paper.** The protocol above is the one specified in the paper's Methodology section (§III.A, *Data Split Protocol*): a *"Stratified Subject-Inclusive protocol for training… which included data from all participants"*, with testing on *"temporally distinct sessions that were entirely excluded from the training process."* That is what this code implements, and §III.A is the description to use when reading this repository alongside the paper. Passages elsewhere in the paper describe the evaluation as subject-independent; the split implemented here is session-held-out, and the 81.56% figure should be read accordingly.

**The headline baseline comparison is not like-for-like.** The +25.02% and +30.62% margins reported in the paper compare the *calibrated* model (99.28%, scored on half the external data) against *uncalibrated* baselines scored on all of it. Under an identical protocol the comparison is **74.26% (SVM) vs 81.56% (base CNN-LSTM), a margin of 7.30 points**. That smaller margin is the one that supports preserving temporal structure rather than flattening it. Both figures appear in the notebook so the distinction is explicit.

**Figures produced by this code.** Where a value in the paper differs from what the notebook prints, the notebook's value is what this implementation produces:

- **Dataset size.** The files load as **188,717 rows (training)** and **101,674 rows (external)** — about 290,000 in total, segmented into 7,256 and 3,909 windows respectively.
- **Feature width.** The pipeline produces **29** features: 19 retained numeric source columns plus 10 derived channels. This is the value consistent with the 1,508-element flattened vector used for the baselines (52 × 29 = 1,508).
- **Preprocessing implemented.** Z-score normalisation, 52-sample windows at 50% overlap, jerk and magnitude. The moving-average smoothing and the 1 s comparative window listed in the paper's Table II are **not** implemented in this code.
- **Confusion matrices.** The external matrix corresponds to the single seed-42 run that scores 81.56%. The 80.53% figure is the mean of the five robustness runs in section 10.

---

## Known issues in this implementation

Disclosed rather than silently patched, because fixing them would change the reported numbers:

1. **Windowing crosses recording boundaries.** The sliding window indexes the concatenated dataframe and does not reset at participant or session boundaries. A small number of windows therefore straddle two recordings and receive the majority (mode) label. Marginal across ~7,200 windows, but incorrect.
2. **Mixed sampling rates under a fixed-sample window.** Recordings were captured at both 20 Hz and 100 Hz; the window is 52 *samples*. It spans ~2.6 s of 20 Hz data but ~0.52 s of 100 Hz data. No resampling to a common rate is performed.
3. **Class mapping is inferred, not stored.** Integer labels are mapped to movement names by ordering classes by frequency, verified once against the source recordings. If the class distribution changed — a different split, a dropped participant — the labels would swap silently and every confusion matrix would be mislabelled without raising an error. Replace this with an explicit mapping before reusing the code.
4. **The 29 retained features have not been enumerated.** `select_dtypes(include=[np.number])` admits every numeric column that survives the drop list. Print `X_raw.columns.tolist()` and confirm no identifier, session index or label-adjacent field is among them.
5. **Asymmetric drop lists.** The training and external-validation cells use slightly different `cols_to_drop` lists. They coincidentally yield the same 29 columns on these files; a column present in one file and not the other would silently change the feature width.

---

## Data

**The data is not in this repository and cannot be redistributed.** The recordings come from human participants under informed consent that does not extend to public release. The code is published for the method.

To run the notebook, place two CSV files in a `data/` directory:

```
data/
├── train_cleaned.csv
└── test_cleaned.csv
```

Expected schema — column names are those written by the logging device and are reproduced verbatim, **including the inconsistent spellings**:

| Column | Notes |
|---|---|
| `AccelroX`, `AceelroY`, `AceelroZ` | Tri-axial accelerometer. Note `Accelro` vs `Aceelro` — this is how the device wrote them and the code matches it. |
| `DMRotX`, `DMRotY`, `DMRotZ` | Tri-axial gyroscope (CoreMotion rotation rate). |
| `MoveType_encoded` | Integer target: one of `{0, 1, 2}`. |
| `MoveType`, `Target`, `Side`, `Wrist`, `Timestamp`, `Unnamed: 0`, `DMUAccel*` | Dropped before training. |
| *(remaining numeric columns)* | Retained as features — see known issue 4. |

Derived channels (`Accel_Mag`, `Gyro_Mag`, and `*_Jerk`) are computed by the notebook; they should not be present in the input.

---

## Running it

```bash
git clone https://github.com/<your-username>/cnn-lstm-imu-motion-recognition.git
cd cnn-lstm-imu-motion-recognition
pip install -r requirements.txt
jupyter notebook hybrid_cnn_lstm_imu.ipynb
```

Then run the cells in order. Global seeds are fixed (`seed_value = 42`, plus `PYTHONHASHSEED`, `TF_DETERMINISTIC_OPS`, `TF_CUDNN_DETERMINISTIC`). The published run used **TensorFlow 2.19.0**, which `requirements.txt` pins, since exact reproduction of the reported accuracies depends on it. The published run was executed on **CPU, with no GPU** (`GPU Available: False` in the section 1 output). Baseline training takes about 91 s; the 5-run robustness loop in section 10 is the slowest part at roughly 110 s per run, so the notebook runs end to end in well under fifteen minutes on an ordinary machine.

---

## Repository contents

```
hybrid_cnn_lstm_imu.ipynb   Full pipeline: preprocessing → training → external
                            validation → transfer learning → baselines →
                            5-run robustness check
requirements.txt            Dependencies
README.md                   This file
data/                       Not included — see Data above
```

The notebook runs in eleven numbered sections and carries its outputs and figures as executed, so it can be read without being run.

---

## Method summary

**Preprocessing.** Z-score normalisation fitted on the training data only and reused unchanged at test time, which is what a fixed deployed device would do. Sliding windows of 52 samples with 50% overlap. Derived channels: accelerometer and gyroscope magnitude, and first differences (jerk).

**Architecture.**

```
Conv1D(64, k=3, relu) → Conv1D(64, k=3, relu) → Dropout(0.5) → MaxPooling1D(2)
  → LSTM(100) → Dropout(0.5)
  → Dense(100, relu) → Dense(3, softmax)
```

Adam, categorical cross-entropy, batch size 64, up to 50 epochs. Class weights computed inversely proportional to class frequency, since wrist pronation is the minority class. Early stopping on validation loss with `patience=5` and `restore_best_weights=True`.

**Calibration (transfer learning).** The external set is split 50/50, stratified by class. The convolutional layers are frozen; the LSTM and dense layers are retrained on the calibration half at a learning rate of 1e-4 for up to 10 epochs with `patience=3`. The reported 99.28% is measured on the other half, which is never seen during fine-tuning.

*Caveat:* that split is random, so calibration and evaluation windows may come from the same session. This matches the intended deployment — a user calibrates, then uses the device — but it is not a session-independent estimate.

---

## Limitations

- Three participants. Two with healthy-range movement, one (`u05`) with an upper-limb impairment. This is not a clinical cohort.
- No subject-independent evaluation. See [above](#what-the-evaluation-does-and-does-not-show).
- Pathological movement — tremor, spasticity, compensatory patterns — is largely unrepresented. Performance on stroke or orthopaedic patients is unverified.
- Controlled collection environment. Sensor slippage, ambient vibration and non-task movement are not modelled.
- A post-calibration accuracy above 99% is promising for unsupervised telerehabilitation but is not evidence of clinical readiness.

## Future work

- Leave-one-subject-out validation on a larger cohort, so generalisation to unseen people is measured rather than assumed.
- Resampling to a common rate before windowing, and windowing within participant/session groups.
- Clinical validation with stroke or orthopaedic patients.
- Edge deployment via TensorFlow Lite on microcontroller-class hardware.
- Unsupervised domain adaptation, to remove the need for a labelled calibration session.

---

## Citation

```bibtex
@inproceedings{eshetu_hybrid_cnn_lstm,
  title     = {Hybrid {CNN}--{LSTM} Framework for Upper-Limb Motion
               Recognition Using Wearable {IMU} Sensors},
  author    = {Eshetu, Sifana Habtamu and Araya, Saron Tesfaye and
               Balcha, Abel Bekele and Damtie, Amanuel Adissu and
               Endale, Amanuel Mulugeta and Benachour, Yassine},
  year      = {2026}
}
```

## Contact

Sifana Habtamu Eshetu — [ORCID 0009-0008-7537-3959](https://orcid.org/0009-0008-7537-3959) · [Google Scholar](https://scholar.google.com/citations?user=MTokPWMAAAAJ)

Corresponding author: Dr. Yassine Benachour, Division of Engineering Technology & Science, Higher Colleges of Technology, Dubai.

## License

Code released under the MIT License. The dataset is not released.
