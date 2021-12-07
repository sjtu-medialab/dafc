# ICMEW:A_Generative_Compression_Framework_For_Low_Bandwidth_Video_Conference

This repository contains the source code for the generation based video compression method described in the paper:
> **A_Generative_Compression_Framework_For_Low_Bandwidth_Video_Conference**  
> *IEEE International Conference on Multimedia & Expo Workshops (ICMEW)*  
> Feng, Y. Huang, Y. Zhang, J. Ling, A. Tang and L. Song  
> [Paper](https://ieeexplore.ieee.org/abstract/document/9455985)
>
> **Abstract:** *Video conferences introduce a new scenario for video transmission, which focuses on keeping the fidelity of faces even in the low bandwidth network environment. In this work, we propose VSBNet, one of the frameworks to utilize face landmarks in video compression. Our method utilizes the adversarial learning to reconstruct origin frames from the landmarks. To recover more details and keep the consistency of identity, we propose the concept of visual sensitivity to separate the contour of the face from the fast-moving parts, such as eyes and mouth. Experimental results demonstrate the superiority of our framework with a low bit rate of around 1KB/s.*
## Important note
**THE METHODS PROVIDED IN THIS REPOSITORY ARE NOT TO BE USED FOR MALICIOUS OR INAPPROPRIATE USE CASES.**  
We release this code in order to help facilitate research of technical counter-measures for detecting this
kind of forgeries. Suppressing this kind of publications will not stop their development but will only make
it more difficult to detect them. 

Please note this is a work in progress, while we make every effort to improve the results of this method, not
every pair of faces can produce a high quality face swap.


## Requirements
- High-end NVIDIA GPUs with at least 11GB of DRAM.
- Either Linux or Windows. We recommend Linux for better performance.
- CUDA Toolkit 10.1, CUDNN 7.5, and the latest NVIDIA driver.
- Python 3.6+ and PyTorch 1.4.0+.

## Data preparation and train
Download the [RAVDESS] (used in the paper) or alternatively any other source of high resolution videos will be fine as well.

The videos should be placed in a flat directory structure.

Run the following command to train all the videos:
```Bash
cd dafc
python train_large_v2.py
```
