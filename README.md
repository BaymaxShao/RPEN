# A Multiple Feature Learning Framework for Real-time Pose Estimation of Nasal Endoscope from Endoscopic Video
This is the official impementation of the paper 'A Multiple Feature Learning Framework for **R**eal-time **P**ose **E**stimation of **N**asal Endoscope from Endoscopic Video'. In this work, a novel deep learning framework is proposed to perform relative pose estimation and absolute pose tracking for nasal endoscope based on endoscopic videos, in which multiple features from two adjacent frames of endoscopic videos are extracted and integrated.

The average error of the localization based on the proposed model is **3.86mm** and the average error of direction prediction is **2.55 degree**, while the relative translation error and rotation error are **0.43mm** and **0.2 degree** in average. In addition, the average inference speed of the proposed framework is faster than 40fps, which achieves the real-time requirement.

- :heavy_check_mark: The visualized results of our work can be found in [our project page](https://rpen-bmxs.netlify.app/).
- :heavy_check_mark: The visualized results of the dataset can be found here.
- :black_square_button: The whole code and dataset will be public when the paper :page_with_curl: is published.

## Implementation Details
OS: **Ubuntu 22.04**

Main Requirements: **Pytorch 1.13.1 + CUDA 11.7**

## Dataset
The examples from the dataset can be found in [here](/data). The structure of the dataset is shown as below:
![image](https://github.com/BaymaxShao/RPEN/assets/91866296/5c526c75-700c-4ae7-a039-401d4eba634a)

The visualizations of 10 samples from the dataset are shown below:

<img src="/data_vis/1.gif" width="400px"> <img src="/data_vis/2.gif" width="400px">

<img src="/data_vis/3.gif" width="400px"> <img src="/data_vis/4.gif" width="400px">

<img src="/data_vis/5.gif" width="400px"> <img src="/data_vis/6.gif" width="400px">

<img src="/data_vis/7.gif" width="400px"> <img src="/data_vis/8.gif" width="400px">

<img src="/data_vis/9.gif" width="400px"> <img src="/data_vis/10.gif" width="400px">

