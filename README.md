# README

## Introduction

We propose Freq-DETR, a frequency-aware real-time transformer detection framework for small object detection in UAV imagery. Frequency-Enhanced Convolution Module extracts features from dual domains to effectively fuse global frequency information with local spatial features. Decoupled Frequency-Domain Feature Interaction Module  decouples high- and low-frequency feature interactions at the intra-scale level while reducing computational overhead. Attention-Guided Selective Feature Pyramid Network employs an attention mechanism to selectively filter and integrate multi-scale features.
Our code is based on Ultralytics. Currently, this file contains the main code for three core modules proposed. We will reiterate our firm commitment to making the complete, cleaned, and executable source code for this project, along with the trained model weights, publicly available on a GitHub repository subsequently.

---

## Contents of the Archive

This archive contains two main directories:

1.  `/ultralytics`
2.  `/experiment logs`

### 1. `/ultralytics` Directory

This directory contains part of the core Python source code for the three novel modules proposed in our paper, which are located in [Freq-DETR/ultralytics/ultralytics/nn/modules](https://github.com/chan0908/Freq-DETR/tree/main/ultralytics/ultralytics/nn/modules) . These files are provided to allow for a direct inspection of the implementation details of our key contributions.

* **`block.py`**: This file contains the implementation of our Frequency-Enhanced Convolution Module (FECM) and Attention-Guided Selective Feature Pyramid Network(AGS-FPN).
* **`DSCEncoder.py`**: This file contains the implementation of our Decoupled Frequency-Domain Feature Interaction Module (DSC-Clo Block). It includes the EfficientAttention and EfficientBlock classes that form our modified hybrid encoder.

### 2. `/experiment logs` Directory

This directory contains a sample, unaltered training log from one of our key experiments.

* **`result-log.csv`**: This is a CSV log file from the complete training run of Freq-DETR model on the VisDrone dataset. The log documents the entire training process , showing the progression of key metrics including:
    * `train/box_loss`, `train/cls_loss`: Training losses, showing a smooth convergence.
    * `metrics/mAP50(B)`, `metrics/mAP50-95(B)`: Validation mAP scores for each epoch, documenting the gradual improvement of the model's performance. The final values in this log directly correspond to the results reported in our manuscript.
    * `lr/pg0`, `lr/pg1`, `lr/pg2`: The learning rate schedule during training.

This log file serves as direct evidence of the experimental process and provides a transparent record of how our final results were achieved.

---
