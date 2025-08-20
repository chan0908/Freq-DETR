# README

## Introduction

This archive is provided as supplementary material to our manuscript, "Freq-DETR: Frequency-Aware Transformer for Real-Time Small Object Detection in Unmanned Aerial Vehicle Imagery." It is intended to address the reviewer's request for materials to aid in the verification and reproduction of our experimental results, and to provide concrete evidence of our research process.

We are confident that these materials demonstrate the authenticity of our work and the implementation of our proposed methods.

---

## Contents of the Archive

This archive contains two main directories:

1.  `/ultralytics`
2.  `/experiment logs`

### 1. `/ultralytics` Directory

This directory contains the core Python source code for the three novel modules proposed in our paper. These files are provided to allow for a direct inspection of the implementation details of our key contributions.

 <font color="blue">All the core codes are located in Freq-DETR/ultralytics/ultralytics/nn/modules.</font>


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

## Commitment to Full Reproducibility

We hope this supplementary material fully addresses the reviewer's concerns. We reiterate our firm commitment to making the complete, cleaned, and executable source code for this project, along with the trained model weights, publicly available on a GitHub repository immediately upon acceptance of the manuscript.

Sincerely,

The Authors