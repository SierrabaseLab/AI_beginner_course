## Introduction : jetson-inference

**Jetson Inference** is an opensource to introduce Deep Neural Network models, which can run in real-time and accurately. Especially, this repository uses NVIDIA TensorRT for efficiently deploying neural networks onto the embedded Jetson platform(e.g. nano, tx2, xaiver ...), improving performance and power efficiency using graph optimizations, kernel fusion, and FP(Float Precision)16/INT8 precision. In more details, refer to [this site(github)](https://github.com/dusty-nv/jetson-inference).

Morerover, it supports to

- training (Transfer Learning / Re-training)
	- Classification
		- Cat/Dog Dataset
		- PlantCLEF Dataset
		- Your Own Image Dataset
	- Object Detection
		- SSD-Mobilenet Network
		- Your Own Detection Dataset
- Inference
	- Classification
		- Imagenet with Image / Video
		- your Own Image
	- Object Detection
		- Face
		- COCO containing dogs, bottles, etc.
	- Semantic Segmentation
		- Cityspcapes Dataset
		- DeepScene Dataset etc.


## 1. Build models

Note that you need to get ready for download opensource.

1. Go into ```AI_beginner_course/DL_course/``` and download opensource.

	```shell
	$ cd ..
	$ git clone https://github.com/dusty-nv/jetson-inference
	$ cd jetson-inference
	$ git submodule update --init
	```

2. Set some settings for build.

	```shell
	$ mkdir build
	$ cd build
	$ cmake ../
	```

	You might be confused because of the options of cmake. You can pass all (That is, Just press 'enter'), except for setup 'pytorch'. In this case, I also recommend to **install pytorch**. 

	It will take a lot of time( ~ 20 mins).

3. Let's build and compile!

	```shell
	$ make -j$(nproc)
	$ sudo make install
	$ sudo ldconfig
	```

## 2. Let's Run some files

1. Here is a files which trained imagenet Datasets by GoogLeNet.

	Image Classification - ImageNet - GoogLeNet

	```shell
	$ cd aarch64/bin/
	$ python3 imagenet.py /dev/video0
	```
