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

The **UBFC‑RPPG** dataset is **not included** in this repository. We are going to use DATASET_1. Please obtain it from the official page:

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

# B-Comparative Performance Analysis of Five Methods

Each evaluation compares five pulse‑measurement methods—**POS**, **GREEN**, **DeepPhys**, **DeepPhysEnhanced**, and **DeepPhysEnhancedReLu**—across four metrics: mean absolute error (MAE), root mean square error (RMSE), Pearson correlation with the reference signal, and signal‑to‑noise ratio (SNR, in dB). Below, we restate the key observations for each evaluation, highlighting the best performance per metric and the results achieved by **DeepPhysEnhanced** (the enhanced deep‑learning method) to indicate its rank.

## Evaluation 5

**GREEN** delivers the best performance for MAE, RMSE, and SNR, while **DeepPhysEnhancedReLu** achieves the highest Pearson correlation. In this context, **DeepPhysEnhanced** reports MAE 21.79 (3rd out of 5), RMSE 26.26 (2nd), Pearson correlation 0.211 (2nd), and SNR 9.39 dB (2nd).

## Evaluation 6

**POS** provides the lowest errors (best MAE and RMSE) and the best SNR, while **DeepPhysEnhancedReLu** attains the strongest Pearson correlation for this sequence. **DeepPhysEnhanced** obtains MAE 21.60 (3rd), RMSE 26.45 (3rd), but its Pearson correlation is only −0.061 (last place, 5th) and its SNR is 9.92 dB (3rd).

## Evaluation 7

**DeepPhysEnhanced** clearly dominates all indicators for this sequence: it achieves the lowest MAE and RMSE, the highest correlation, and the best SNR. Specifically, **DeepPhysEnhanced** reaches MAE 24.18 (1st out of 5), RMSE 31.04 (1st), Pearson correlation 0.143 (1st), and SNR 9.69 dB (1st).

## Evaluation 8

**GREEN** presents the best error values (lowest MAE and RMSE) as well as the best SNR, while **POS** achieves the highest Pearson correlation. Under these conditions, **DeepPhysEnhanced** attains MAE 13.54 (3rd), RMSE 16.87 (3rd), Pearson correlation 0.140 (4th), and SNR 12.14 dB (3rd).

## Evaluation 10

**GREEN** wins on MAE, RMSE, and SNR, while **DeepPhysEnhanced** secures the best Pearson correlation on this sequence. **DeepPhysEnhanced** records MAE 13.23 (2nd), RMSE 20.41 (2nd), Pearson correlation 0.238 (1st), and SNR 11.00 dB (2nd).

## Evaluation 11

**POS** outperforms the others across all metrics in this evaluation (best MAE, RMSE, correlation, and SNR). **DeepPhysEnhanced** achieves MAE 21.34 (3rd), RMSE 27.65 (3rd), Pearson correlation 0.213 (2nd), and SNR 8.98 dB (3rd).

## Evaluation 12

**GREEN** delivers the best performance on every metric for this sequence (lowest MAE and RMSE, highest correlation, and best SNR). **DeepPhysEnhanced** reports MAE 31.88 (3rd), RMSE 35.84 (3rd), Pearson correlation 0.084 (2nd), and SNR 8.43 dB (3rd).

## Evaluation 13

This sequence is an atypical case where the original deep‑learning method **DeepPhys** has the advantage on all metrics, obtaining the lowest MAE/RMSE, the highest correlation, and the best SNR. In comparison, **DeepPhysEnhanced** records MAE 44.03 (3rd), RMSE 52.69 (3rd), Pearson correlation −0.209 (very low, 5th), and SNR 6.76 dB (3rd).

## Overall Synthesis of the Evaluations

Across all eight evaluations, two methods stand out for their overall performance. **GREEN** emerges as the most accurate and stable method, winning the largest number of first places in error (4 wins in MAE and 4 in RMSE) as well as in SNR (4 wins). **POS** appears as a robust alternative, with 2 wins in MAE/RMSE and 2 in SNR. The raw deep‑learning approaches—particularly **DeepPhys**—lag overall: aside from Evaluation 13, where **DeepPhys** exceptionally surpasses the others on all four metrics, these learned methods do not match the reliability of the classical approaches on most tested sequences.

The **DeepPhysEnhanced** method (a DeepPhys network enriched with **POS** and **GREEN** channels) typically ranks among the front‑runners without dominating consistently: it frequently posts intermediate ranks (often 2nd or 3rd) in terms of error (MAE/RMSE) and SNR, while achieving the best correlations in two cases (Evaluations 7 and 10). This underscores its potential for fine temporal tracking of the pulse (fidelity of the waveform), even though its amplitude errors and noise remain slightly higher than the best classical methods overall.

These observations are consistent with aggregated averages computed over all eight evaluations. On average, **GREEN** shows the lowest errors (mean MAE 19.56 and mean RMSE 25.31) and the best mean SNR (11.48 dB). **POS** follows closely with a mean MAE of 23.01, mean RMSE of 28.30, and mean SNR of 10.23 dB. **DeepPhysEnhanced** exhibits average errors comparable to **POS** (MAE 23.95; RMSE 29.65) but a slightly lower SNR (9.54 dB). In terms of mean correlation with the reference signal, **POS**, **GREEN**, and **DeepPhysEnhanced** obtain similar values (around 0.09–0.12), whereas **DeepPhys** is weaker (0.041). The **DeepPhysEnhancedReLu** variant stands out with the highest mean correlation (0.123), but this comes with higher error values (mean MAE 30.44) and a lower SNR (7.40 dB), suggesting a trade‑off in which improved temporal tracking is paid for with reduced accuracy and increased noise.

## DeepPhysEnhanced: Method‑Specific Analysis

**DeepPhysEnhanced** deserves special attention, as it is designed as an extension of the original **DeepPhys** model that explicitly integrates channels derived from the classical approaches (**POS** and **GREEN**). This fusion allows **DeepPhysEnhanced** to benefit from more stable photoplethysmographic signals (thanks to the POS/GREEN channels) while retaining certain attributes of the learned model (ability to model complex relations, but also some variability under unfavorable conditions).

In concrete terms, **DeepPhysEnhanced** often delivers intermediate‑to‑high performance: on many sequences, it ranks 2nd or 3rd for MAE, RMSE, and SNR. Its added value appears mainly in temporal tracking (correlation): it showed marked gains in Pearson correlation in evaluations such as **7** and **10**, where it outperformed all reference methods by closely following the cardiac signal’s temporal oscillation. When the additional POS/GREEN channels are informative and consistent with the true pulse dynamics (e.g., low subject motion, good illumination, limited artifacts), the learned model leverages them to outperform classical baselines, as seen with its clean sweep in Evaluation 7 and its top correlation in Evaluation 10.

Conversely, when these additional channels are perturbed (significant subject movement, specular reflections or lighting changes, diverse skin tones introducing color‑channel noise), **DeepPhysEnhanced** may assign weight to misleading signals. In such cases, the model’s predictions become less reliable: we observe very low or even negative correlation with the reference signal (e.g., Evaluations 6 and 13), reflecting erratic temporal tracking, often accompanied by a simultaneous SNR degradation. This behavior highlights an important limitation: the performance of **DeepPhysEnhanced** varies more with scene content and input‑signal quality, which can introduce instability when conditions deviate from the optimal ones.

In practice, the **DeepPhysEnhanced** variant (combining deep learning with expert POS/GREEN channels) is particularly suitable for sequences where the rPPG component is pronounced and minimally contaminated by motion noise. For more challenging ("adversarial") pulse‑measurement content—such as high activity or changing lighting—it is advisable to add safeguards when using **DeepPhysEnhanced**: for example, detect when correlation becomes negative or abnormally low, monitor SNR levels, and automatically fall back to a more reliable method like **GREEN** or **POS**. Such hybrid strategies would help secure signal exploitation by leveraging the best of both worlds—the temporal fidelity provided by deep learning when possible, and the stability of classical methods when in doubt.

## Conclusion of the Evaluations (General Overview)

Based on all results across the eight evaluations, several major lessons emerge. **GREEN** establishes itself as the versatile reference, offering the best overall accuracy (minimal errors) and the best SNR in most cases, making it a strong default choice for estimating pulse from video. **POS** is a reliable second choice: its average performance is slightly below **GREEN**, yet it outperforms all deep‑learning methods in most situations and even surpasses **GREEN** on a few specific sequences (it dominated all metrics in Evaluation 11, for example).

The deep model **DeepPhys** (without additional channels) appears less competitive on this test set, delivering strong results only in a very particular case (Evaluation 13). This suggests that without incorporating prior knowledge (dedicated color channels) or without test conditions aligned with its training, this model suffers from generalization or robustness issues under varying conditions.

The **DeepPhysEnhanced** method, which enriches **DeepPhys** with **POS/GREEN** channels, demonstrates real potential for pulse measurement: in some cases, it can offer the best temporal tracking (maximizing correlation with the true pulse) while maintaining attractive trade‑offs between error and noise. However, its performance is not uniform and depends more on scene content (capture conditions, motion, etc.), which introduces variability. The **DeepPhysEnhancedReLu** variant illustrates this trade‑off: by modifying the output activation, it can sometimes further improve correlation (capturing instantaneous heart‑rate dynamics better), but at the cost of degraded stability (higher errors and lower SNR), indicating greater difficulty in maintaining absolute accuracy.

## Usage Recommendation

For general use and application contexts where overall reliability is paramount, it is recommended to favor **GREEN**, given its robustness and excellent average results. **POS** can also be used as an alternative or complement, being conceptually simple and inexpensive while offering performance close to **GREEN**. In contrast, deep‑learning‑based methods should be approached with greater caution. **DeepPhysEnhanced** can become a serious candidate in scenarios where the temporal fidelity of the signal is crucial (e.g., precise tracking of heart‑rate variability) and where the quality of the input channels can be controlled (limited motion, good illumination, etc.). Under these optimal conditions, it has shown that it can surpass classical methods on certain aspects of the signal. Nevertheless, it is strongly advised to integrate safeguards when using it in practice: for example, define confidence thresholds on correlation and SNR to decide whether the measurement is usable, and plan automatic fallback strategies to traditional methods (**GREEN**/**POS**) when the signal is questionable. Furthermore, future work could focus on improving the robustness of **DeepPhysEnhanced** (e.g., augmented training to better handle motion artifacts or adding adaptive filters) to reduce its variability and make it more reliable under diverse conditions. As the results currently stand, **GREEN** remains the reference method for stable, all‑around performance, **POS** provides a strong baseline in second place, and **DeepPhysEnhanced** represents a promising avenue for pushing beyond, provided its limits are managed through careful, guarded use.

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
