# StairNet series——Deep Learning Real-time Stair Detection(StairNet-Depth-In)

StairNet series are a state-of-the-art algorithms
that uses deep convolutional neural networks to perform stair detection.
According to the input modality, StairNet series have two kinds of architectures 
including StairNet-Depth-In and StairNet-Depth-Out. StairNet-Depth-In has two input
 branches of RGB and Depth images, which makes it can even work in dark night.
## Demo of landing point detection with [realsense D435i](https://www.intelrealsense.com/depth-camera-d435i/)
<img src=".\materials\landing_points_detection.gif" width="640" height="360"/>

## Demo of stair measurement with [realsense D435i](https://www.intelrealsense.com/depth-camera-d435i/)
<img src=".\materials\geometric_parameter_estimation.gif" width="640" height="360"/>

## StairNet-Depth-In
<img src="https://github.com/MrChenWang/StairNet-DepthIn/blob/main/materials/Depth_In_arc.png" width="" height=""/>

## Getting Started
To get started, we recommend Anaconda for people using a GPU, and please install the proper dependencies as shown below:
```//conda
conda create -n stairnet python=3.7
conda activate stairnet
```
```//conda
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python
pip install tqdm
```
## Downloading Dataset
Our dataset is stored in Mendeley Data which has 2276 RGB-D image pairs as training set and 556 RGB-D image pairs as validation set.
All the images are padded and scaled to 512 × 512.

The original labels have the following format:

cls x1 y1 x2 y2

where cls represents the class of stair lines, x1 and y1 represent the left endpoints, and x2 and y2 represent the right endpoints, you need use the **label_transfer.py** to get the final segmented labels. The dataset is linked below:

[https://data.mendeley.com/datasets/6kffmjt7g2/1](https://data.mendeley.com/datasets/6kffmjt7g2/1)
## Downloading Weights
We provide 3 models for StairNet including StairNet 1.0 ×, StairNet 0.75 × and StairNet 0.5 × to meet various detection scenes. We use a width-factor to multiply by the channels to adjust the model parameters. The weights are linked below:

[Baidu Netdisk](https://pan.baidu.com/s/1oQQD4dDhMxGmSvC2o87X_w?pwd=w95o)

## Visualization results

<img src=".\materials\results.png"/>

## Citation
If our method and dataset are useful to your research, please consider to cite us:
```//Latex
@article{stairnetv1,
    author = {Chen Wang and Zhongcai Pei and Shuang Qiu and Zhiyong Tang},
    title = {Deep leaning-based ultra-fast stair detection},
    journal = {scientific reports},
    volume = {12},
    year = {2022},
    article-number = {16124},
    doi = {10.1038/s41598-022-20667-w}
}
```
```//Latex
@article{stairnetv2,
    author = {Chen Wang and Zhongcai Pei and Shuang Qiu and Zhiyong Tang},
    title = {RGB-D-Based Stair Detection and Estimation Using Deep Learning},
    journal = {Sensors},
    volume = {23},
    year = {2023},
    number = {4},
    article-number = {2175},
    doi = {10.3390/s23042175}
}
```
```//Latex
@article{stairnetv3,
    author = {Chen Wang and Zhongcai Pei and Shuang Qiu and Yachun Wang and Zhiyong Tang},
    title = {StairNetV3: depth-aware stair modeling using deep learning},
    journal = {The Visual Computer},
    year = {2024},
    doi = {10.1007/s00371-024-03268-8}
}
```
```//Latex
@misc{rgbd_stair_dataset,
    author = {Chen Wang and Zhongcai Pei and Shuang Qiu and Yachun Wang and Zhiyong Tang},
    title = {RGB-D stair dataset},
    journal = {Mendeley Data},
    year = {2023},
    doi = {10.17632/6kffmjt7g2.1}
}
```

## License
This project is licensed under the [Creative Commons NonCommercial (CC BY-NC 3.0)](https://creativecommons.org/licenses/by-nc/3.0/) 
license where only non-commercial usage is allowed. For commercial usage, please contact us.
## Future works

We will improve and add inference scripts for webcam and realsense D435i depth camera in the future.

## Contact
If you have any questions or suggestions, feel free to contact us at this email: venus@buaa.edu.cn
