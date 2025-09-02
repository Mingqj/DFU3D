# Enhancing Pseudo-Boxes via Data-Level LiDAR-Camera Fusion for Unsupervised 3D Object Detection

<div align="center">

[Mingqian Ji](https://github.com/Mingqj) </sup>,
[Shanshan Zhang](https://shanshanzhang.github.io/) ✉</sup>,
[Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN) </sup>

PCA Lab, School of Computer Science and Engineering, Nanjing University of Science and Technology

✉ Corresponding author

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2508.20530)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

## 📖 About

This repository represents the official implementation of the paper titled "Enhancing Pseudo-Boxes via Data-Level LiDAR-Camera Fusion for Unsupervised 3D Object Detection".

We propose a novel unsupervised 3D object detection framework built on early-stage data-level fusion of LiDAR and RGB images. Specifically, we design a bi-directional fusion module, where LiDAR points inherit semantic category labels from 2D instance segmentation, while image pixels are projected into 3D to densify sparse point clouds. To address errors from depth estimation and segmentation, we introduce a noise suppression module that applies local radius filtering to reduce depth noise and global statistical filtering to remove outliers. Finally, a dynamic self-evolution module iteratively refines pseudo-boxes under dense representations, leading to more accurate and reliable detection results.

![](./resources/pipeline.png)

## 💾 Main Results

**nuScenes val set**
| Methods     | Label  | Vehicle | Pedestrian | Cyclist|   All    |
|:------------:|:----:|:----:|:---:|:---:|:---:|
| Supervised  | 1% | 39.3 | 31.8| 1.8 | 14.3|
| DFU3D  | 0 | 32.3 | 37.7| 15.3 | 28.4|

**KITTI set**
| Methods    | Label | Easy | Mod. | Hard |
|:------------:|:----:|:----:|:---:|:---:|
|Supervised|    100%   | 97.1 | 89.2 | 81.8|
|DFU3D|    0   | 95.1 | 97.3 | 81.0|
