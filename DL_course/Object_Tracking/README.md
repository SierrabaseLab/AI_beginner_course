# Unsupervised Deep Tracking, UDT

1. Introduction
2. Settings
3. Implementation

## 1. Introduction

There is the [paper](https://arxiv.org/abs/1904.01828), UDT. and the original open source of this paper is here 
~~~
C/C++ version (Original) : https://github.com/594422814/UDT
Python(pytorch) version  : https://github.com/594422814/UDT_pytorch
~~~

From this repository, I modified [UDT_pytorch](https://github.com/594422814/UDT_pytorch)

And ,there is a his citation;
```
@inproceedings{Wang_2019_Unsupervised,
    title={Unsupervised Deep Tracking},
    author={Wang, Ning and Song, Yibing and Ma, Chao and Zhou, Wengang and Liu, Wei and Li, Houqiang},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}

@article{wang17dcfnet,
    Author = {Qiang Wang, Jin Gao, Junliang Xing, Mengdan Zhang, Weiming Hu},
    Title = {DCFNet: Discriminant Correlation Filters Network for Visual Tracking},
    Journal = {arXiv preprint arXiv:1704.04057},
    Year = {2017}
}
```

## 2. Settings
1. System Environments
	~~~
	Ubuntu == 18.04
	Cuda == 10.1
	Cudnn == 7.6.5
	Python == 3.6.5
	~~~
2. Python Library
	~~~
	torch == 1.4.0
	torchvision == 0.5.0
	numpy == 1.18.1
	opencv-contrib-python == 4.1.0.25
	~~~
## 3. Implementation
1. You can just run this file ```UDT.py```

	```
	$ python3 UDT.py
	```

	If the camera recognized in computer, this file will be operated well.

2. You can change the argument options ```--input_sources```

	1. Web Camera (USB Camera)
		
		Use ```--input_source /dev/video0```, ```--input_source 0```
		
		Also, you can use ```--input_sourcve=/dev/video0```, ```--input_source=0```

		```shell
		$ python3 UDT.py --input_source /dev/video0
		```

	2. Other Video Source

		Use ```--input_source {your files} ``` or ```--input_source={your_files}```/

		```shell	
		$ python3 UDT.py --input_source bag.avi
		```


