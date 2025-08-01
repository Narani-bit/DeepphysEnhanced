# DeepPhysEnhanced: Robust rPPG Signal Extraction under Motion

This repository contains the implementation and testing pipeline for the **DeepPhysEnhanced** model, an improvement over the original DeepPhys method for remote photoplethysmography (rPPG) under motion conditions.

## Repository Structure

### 1. `DeepPhysEnhanced.ipynb`

* A Google Colab notebook that serves as the full processing **pipeline** for training, testing, and evaluating the DeepPhysEnhanced model.
* Integrates video loading, preprocessing, model inference, and result visualization.

### 2. `DeepPhysEnhanced.py`

* Contains the full **Python implementation** of the DeepPhysEnhanced model.
* Enhances the original DeepPhys with a three-branch architecture.
* Incorporates an attention mechanism to fuse information from all three sources.

### 3. `videos.zip`

* A ZIP archive of **test videos** used to evaluate rPPG estimation performance under various motion conditions.
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

## Comparison of POS and GREEN Methods on the Same Test Data

| **Criterion**              | **DeepPhys**                                                | **POS**                                                                 | **GREEN**                                                     | **DeepPhys Enhanced**                                                               |
| -------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Estimation Accuracy**    | High overall accuracy (e.g., frontal: \~70.6 BPM vs others) | Accurate in some cases but large errors under motion (cam\_move \~46.8) | High error under motion, accurate only in stable conditions   | Better overall consistency, clear improvement under motion (e.g., cam\_move \~71.1) |
| **Signal Quality (rPPG)**  | Attention mechanism captures robust pulsatile components    | Generally good, due to multi-channel combination                        | Average, depends on green channel prone to disturbance        | Multi-source fusion (POS + GREEN + DeepPhys) provides a more stable signal          |
| **Motion Robustness**      | Robust under motion (e.g., face\_move: 75 BPM, stable)      | Weak under motion (face\_move: 121.3 BPM)                               | Very sensitive to motion (cam\_move: 46.8 vs actual \~73 BPM) | Very robust due to integration of 3 sources with attention mechanism                |
| **Complexity & Time**      | Moderate (CNN + attention network)                          | Reasonable complexity                                                   | Very lightweight                                              | Higher complexity (3 branches), still reasonable runtime                            |
| **Ease of Implementation** | Medium (deep network with attention and 2 branches)         | Medium                                                                  | Very easy (single channel only)                               | More complex: requires synchronized inputs from 3 sources                           |

## 📜 Citation

```bibtex
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```
