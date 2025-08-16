# DeepPhysEnhanced: Robust rPPG Signal Extraction under Motion

This repository contains the implementation and testing pipeline for the DeepPhysEnhanced model, an improvement over the original DeepPhys method for remote photoplethysmography (rPPG) under motion conditions.

## Repository Structure

### 1. `DeepPhysEnhanced.ipynb`

A Google Colab notebook that performs the full processing workflow for DeepPhysEnhanced, including:

* Installing dependencies and uploading input files (videos & model script)
* Extracting and preprocessing facial regions using MediaPipe
* Generating training inputs: raw frames, temporal differences, and POS/GREEN features
* Training the DeepPhysEnhanced model using PyTorch
* Performing inference and estimating heart rate (BPM) via FFT from predicted signals

### 2. `DeepPhysEnhanced.py`

Contains the full PyTorch implementation of the DeepPhysEnhanced model.

This model improves upon DeepPhys by using a three-branch architecture to process and fuse:

* Temporal motion differences
* Appearance (raw frames with attention masks)
* POS + GREEN channel features

Key features:

* Attention mechanisms applied at two stages in the appearance stream
* Fusion of all branches before final heart rate estimation
* Designed for robust rPPG prediction even under motion and illumination variations

### 3. `videos.zip`

* A ZIP archive of test videos used to evaluate rPPG estimation performance under various motion conditions.
* Scenarios include:

  * `cam_move.mp4`
  * `dark.mp4`
  * `face_move.mp4`
  * `frontal.mp4`
  * `light.mp4`
  * `light_face_move.mp4`

### 4. `results_csv/`

* A directory containing **CSV files** with the test results for the following models:

  * `POS`
  * `GREEN`
  * `DeepPhys`
  * `DeepPhysEnhanced`
* Each CSV includes BPM estimations across different test conditions for comparison.

### 5. UBFC‑RPPG (external download)

The **UBFC‑RPPG** dataset is **not included** in this repository. Please obtain it from the official page:

* [https://sites.google.com/view/ybenezeth/ubfcrppg](https://sites.google.com/view/ybenezeth/ubfcrppg)

Follow the provider’s terms and instructions on that page.

### 6. `DeepphysEnhanced/`

Artifacts for the tanh-activation variant of DeepPhysEnhanced:

* `DeepPhysEnhanced_ipynb`: a Jupyter notebook covering the entire pipeline (split, train, validation, test, BPM results, BPM per second).
* `bpm_results.csv`: BPM measurements for each video in the UBFC-RPPG dataset.
* `metrics_summary.csv`: evaluation metrics collected during training, validation, and test.
* `loss_history_tanh.csv`: training loss per epoch when using tanh activation.
* `val_loss_history.csv`: validation loss per epoch when using tanh activation.
* `DeepphysEnhanced.py`: model implementation using the tanh activation function.
* `checkpoints/`: contains `deepphys_enhanced_last.pt` and `deepphys_enhanced_weights.pth`.

### 7. `DeepphysEnhancedReLu/`

Artifacts for the ReLU-activation variant:

* `DeepPhysEnhancedReLu_ipynb`: a Jupyter notebook with the full pipeline (split, train, validation, test, BPM results, BPM per second) using ReLU activation.
* `bpm_results.csv`: BPM measurements for each video in the UBFC-RPPG dataset.
* `metrics_summary.csv`: evaluation metrics captured during training, validation, and test.
* `loss_history_ReLu.csv`: training loss per epoch with ReLU activation.
* `val_loss_history_ReLu.csv`: validation loss per epoch.
* `DeepphysEnhancedReLu.py`: model implementation using the ReLU activation function.
* `checkpoints/`: contains `deepphys_enhanced_last.pt` and `deepphys_enhanced_weights.pth`.

### 8. `results_hr/`

Per-second BPM time series for each video and for each model. These files are used for downstream evaluation of model metrics.

### 9. `BPM_metrics.ipnb`

A Jupyter notebook used to compare the `results_hr` outputs across models for every video and to derive evaluation metrics.

### 10. `Evaluation/`

Aggregated evaluation metrics of the models for each video in the dataset.

## A- Comparison of POS and GREEN Methods on the Same Test Data

| **Criterion**              | **DeepPhys**                                                | **POS**                                                                 | **GREEN**                                                     | **DeepPhys Enhanced**                                                               |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Estimation Accuracy**    | High overall accuracy (e.g., frontal: \~70.6 BPM vs others) | Accurate in some cases but large errors under motion (cam\_move \~46.8) | High error under motion, accurate only in stable conditions   | Better overall consistency, clear improvement under motion (e.g., cam\_move \~71.1) |
| **Signal Quality (rPPG)**  | Attention mechanism captures robust pulsatile components    | Generally good, due to multi-channel combination                        | Average, depends on green channel prone to disturbance        | Multi-source fusion (POS + GREEN + DeepPhys) provides a more stable signal          |
| **Motion Robustness**      | Robust under motion (e.g., face\_move: 75 BPM, stable)      | Weak under motion (face\_move: 121.3 BPM)                               | Very sensitive to motion (cam\_move: 46.8 vs actual \~73 BPM) | Very robust due to integration of 3 sources with attention mechanism                |
| **Complexity & Time**      | Moderate (CNN + attention network)                          | Reasonable complexity                                                   | Very lightweight                                              | Higher complexity (3 branches), still reasonable runtime                            |
| **Ease of Implementation** | Medium (deep network with attention and 2 branches)         | Medium                                                                  | Very easy (single channel only)                               | More complex: requires synchronized inputs from 3 sources                           |

## B- Metric Evaluation

### Per‑Evaluation Interpretation

#### Evaluation 5

In this evaluation, the best scores are achieved by **GREEN** for MAE, **GREEN** for RMSE, **DeepPhysEnhancedReLu** for Pearson correlation, and **GREEN** for SNR. **DeepPhysEnhanced** reports MAE 21.79 (rank 3/5), RMSE 26.26 (rank 2/5), Pearson 0.211 (rank 2/5), and SNR 9.39 dB (rank 2/5).

#### Evaluation 6

In this evaluation, the best scores are achieved by **POS** for MAE, **POS** for RMSE, **DeepPhysEnhancedReLu** for Pearson correlation, and **POS** for SNR. **DeepPhysEnhanced** reports MAE 21.60 (rank 3/5), RMSE 26.45 (rank 3/5), Pearson −0.061 (rank 5/5), and SNR 9.92 dB (rank 3/5).

#### Evaluation 7

In this evaluation, the best scores are achieved by **DeepPhysEnhanced** for MAE, **DeepPhysEnhanced** for RMSE, **DeepPhysEnhanced** for Pearson correlation, and **DeepPhysEnhanced** for SNR. **DeepPhysEnhanced** reaches MAE 24.18 (rank 1/5), RMSE 31.04 (rank 1/5), Pearson 0.143 (rank 1/5), and SNR 9.69 dB (rank 1/5).

#### Evaluation 8

In this evaluation, the best scores are achieved by **GREEN** for MAE, **GREEN** for RMSE, **POS** for Pearson correlation, and **GREEN** for SNR. **DeepPhysEnhanced** obtains MAE 13.54 (rank 3/5), RMSE 16.87 (rank 3/5), Pearson 0.140 (rank 4/5), and SNR 12.14 dB (rank 3/5).

#### Evaluation 10

In this evaluation, the best scores are achieved by **GREEN** for MAE, **GREEN** for RMSE, **DeepPhysEnhanced** for Pearson correlation, and **GREEN** for SNR. **DeepPhysEnhanced** records MAE 13.23 (rank 2/5), RMSE 20.41 (rank 2/5), Pearson 0.238 (rank 1/5), and SNR 11.00 dB (rank 2/5).

#### Evaluation 11

In this evaluation, the best scores are achieved by **POS** for MAE, **POS** for RMSE, **POS** for Pearson correlation, and **POS** for SNR. **DeepPhysEnhanced** reports MAE 21.34 (rank 3/5), RMSE 27.65 (rank 3/5), Pearson 0.213 (rank 2/5), and SNR 8.98 dB (rank 3/5).

#### Evaluation 12

In this evaluation, the best scores are achieved by **GREEN** for MAE, **GREEN** for RMSE, **GREEN** for Pearson correlation, and **GREEN** for SNR. **DeepPhysEnhanced** reports MAE 31.88 (rank 3/5), RMSE 35.84 (rank 3/5), Pearson 0.084 (rank 2/5), and SNR 8.43 dB (rank 3/5).

#### Evaluation 13

In this evaluation, the best scores are achieved by **DeepPhys** for MAE, **DeepPhys** for RMSE, **DeepPhys** for Pearson correlation, and **DeepPhys** for SNR. **DeepPhysEnhanced** shows MAE 44.03 (rank 3/5), RMSE 52.69 (rank 3/5), Pearson −0.209 (rank 5/5), and SNR 6.76 dB (rank 3/5).

### Overall Interpretation

Overall trends are clear: **GREEN** leads both accuracy and stability, with 4 wins in MAE, 4 in RMSE, and 4 in SNR across the 8 evaluations, while **POS** is a robust alternative with 2 wins in MAE/RMSE/SNR. **DeepPhys** is generally behind except for a singular case where it tops all four metrics. **DeepPhysEnhanced** most often sits in the upper‑middle of the rankings and secures two first places for Pearson correlation, confirming its potential for temporal tracking.

These findings align with aggregated means: **GREEN** yields the lowest average errors (MAE 19.56, RMSE 25.31) and the best average SNR (11.48 dB), with **POS** close behind. **DeepPhysEnhanced** shows averages near **POS** for MAE/RMSE but slightly lower SNR, and its mean correlation is comparable to **GREEN**. The **DeepPhysEnhancedReLu** variant sometimes improves correlation but at the cost of higher errors and lower SNR, indicating reduced stability.

### Averages across 8 evaluations

| Method               | Mean MAE | Mean RMSE | Mean Pearson | Mean SNR (dB) |
| -------------------- | -------- | --------- | ------------ | ------------- |
| POS                  | 23.01    | 28.30     | 0.117        | 10.23         |
| GREEN                | 19.56    | 25.31     | 0.095        | 11.48         |
| DeepPhys             | 51.73    | 58.17     | 0.041        | 3.45          |
| DeepPhysEnhanced     | 23.95    | 29.65     | 0.095        | 9.54          |
| DeepPhysEnhancedReLu | 30.44    | 36.74     | 0.123        | 7.40          |

### Win count per metric (8 evaluations)

| Method               | #MAE | #RMSE | #Pearson | #SNR |
| -------------------- | ---- | ----- | -------- | ---- |
| POS                  | 2    | 2     | 2        | 2    |
| GREEN                | 4    | 4     | 1        | 4    |
| DeepPhys             | 1    | 1     | 1        | 1    |
| DeepPhysEnhanced     | 1    | 1     | 2        | 1    |
| DeepPhysEnhancedReLu | 0    | 0     | 2        | 0    |

### Focus: DeepPhysEnhanced vs. other methods

**DeepPhysEnhanced** is built on **DeepPhys** and explicitly integrates channels derived from **POS** and **GREEN**. This fusion explains why it retains a noise/error structure close to DeepPhys while benefiting from the more stable PPG signals of POS/GREEN. In practice, this yields frequent 2nd–3rd places in MAE/RMSE and SNR, with marked gains in Pearson correlation in Evaluations 7 and 10. When the added channels are informative and consistent with the true dynamics, the model outperforms classical baselines (a clean sweep in Evaluation 7 and best correlation in Evaluation 10). Conversely, when those channels are perturbed (motion, specularities, skin tone variations, or non‑stationary lighting), the learned weighting can misfire: correlation drops to low or negative values (Evaluations 6 and 13), exposing a risk of unstable temporal tracking.

In practice, the POS/GREEN‑augmented variant is particularly suitable for sequences where the rPPG component is salient and minimally contaminated by motion artifacts. For more adversarial content, adding safety mechanisms (negative‑correlation detection, SNR gating, or automatic fallback to GREEN/POS) secures signal exploitation.

### Conclusion

Across the eight evaluations, **GREEN** emerges as the versatile reference for accuracy and SNR, **POS** is a strong second choice, and **DeepPhys** is competitive only in an atypical case. **DeepPhysEnhanced**—defined as a DeepPhys extension incorporating POS/GREEN channels—shows clear potential: it can deliver the best temporal tracking and attractive error/noise trade‑offs, but its performance varies more with scene conditions. For general use, prefer **GREEN**; for scenarios where temporal fidelity is paramount and channel quality can be controlled, **DeepPhysEnhanced** becomes a strong candidate, ideally paired with safeguards (thresholds on correlation and SNR, plus fallback strategies).

## C- Activation Functions ReLU and Tanh using loss\_history

### Comparative Table of Activation Functions ReLU and Tanh

| Aspect                                     | ReLU                                                                                | tanh                                                                            |
| ------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Start                                      | Starts lower (**0.004784**), initial advantage in absolute loss.                    | Starts higher (**0.026765**), initial handicap later compensated.               |
| Initial speed (epochs 1–10)                | Fast decrease (slope ≈ **−3.63e−4**), 10% of initial loss reached at **epoch 5**.   | Very fast decrease (slope ≈ **−1.52e−3**), 10% reached as early as **epoch 2**. |
| Mid-training                               | Plateaus around **\~1.1e−4** near **epochs \~6–7** and progresses little afterward. | Keeps **decreasing beyond 10** and **crosses 1e−4** at **epoch 16**.            |
| Best point & end                           | Best loss **1.07e−4** at **epoch 27**; final loss **1.14e−4**.                      | Best loss **2.59e−5** at **epoch 24**; final loss **3.55e−5**.                  |
| Curve crossover                            | Stays better **before epoch 15**, then is overtaken.                                | **Drops below ReLU at epoch 15** and **stays** below thereafter.                |
| Cumulative loss (sum over 29 epochs)       | **Lower** (**0.01137**), favorable if training stops early.                         | **Higher** (**0.04050**) due to a high start, but wins at the end of training.  |
| End-of-training stability (last 10 epochs) | Very stable (**σ ≈ 4.22e−6**), minimal variations around the plateau.               | Stable (**σ ≈ 2.90e−6**) despite a **much lower** loss level.                   |

### Interpretation

ReLU offers an excellent start and a low cumulative loss if training stops early, but it plateaus fairly quickly (\~1.1e−4) and does not reach 1e−4 on these data (best at 1.07e−4, final 1.14e−4).

Tanh starts much higher but accelerates more strongly, overtakes ReLU at epoch 15, drops below 1e−4 at epoch 16, and reaches better convergence (best 2.59e−5, final 3.55e−5).

### Conclusion

In our setting (29 epochs), **tanh** is the best for achieving the lowest final loss and deeper convergence. No marked instability was observed: ReLU remains very smooth but plateaued, while tanh shows slight ripples at the end of training, consistent with a very low loss and without problematic oscillations.

### Improvement Suggestions

* **For tanh:** add a learning-rate scheduler; an early stop based on train-loss stagnation is an alternative.

## 📜 Citation

```bibtex
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}

@article{bobbia2017unsupervised,
  title={Unsupervised skin tissue segmentation for remote photoplethysmography},
  author={Bobbia, S and Macwan, R and Benezeth, Y and Mansouri, A and Dubois, J},
  journal={Pattern Recognition Letters},
  year={2017}
}
```
