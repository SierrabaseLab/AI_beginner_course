## Introduction : Jetson-Inference

```Jetson Inference``` is an opensource to introduce Deep Neural Network models, which can run in real-time and accurately. Especially, this repository uses NVIDIA TensorRT for efficiently deploying neural networks onto the embedded Jetson platform(e.g. nano, tx2, xaiver ...), improving performance and power efficiency using graph optimizations, kernel fusion, and FP(Float Precision)16/INT8 precision. In more details, refer to [this site(github)](https://github.com/dusty-nv/jetson-inference).

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

	You might be confused because of the options of cmake. Note that the default models is

	- Image Recoginition
		- GoogleNet
		- ResNet-18
	- Object Detection
		- SSD-MobileNet-v2
		- PedNet
		- FaceNet
		- DetectNet-COCO-Dog
	- Semantic Segmentation
		- FCN-ResNet18-Cityspaces-512x256
		- FCN-ResNet18-DeepScene-576x320
		- FCN-ResNet18-MHP-512x320
		- FCN-ResNet18-Pascal-VOC-320x320
		- FCN-ResNet18-SUN-RGBD-512x400

	**Just press ```enter```**(Recommend). If you want to download all, you can select menu, where contains "all models", by pressing ```spacebar```.

	Moreover, I recommend to select to install ```pytorch```


	It will take a lot of time( ~ 20 mins).

3. Let's build and compile!

	```shell
	$ make -j$(nproc)
	$ sudo make install
	$ sudo ldconfig
	```

## 2. Let's Run some files

Let's check the results of pretrained models.

1. Image Classification - [ImageNet](http://www.image-net.org/)

	Here it is a basic step)

	```shell
	$ cd aarch64/bin/
	$ python3 imagenet.py /dev/video0
	```

	Note that the default option of ```imagenet.py``` file is "network=GoogLeNet". and "/dev/video0" means your data input sources. Espeically, it means you'll try a ```USB viedo Camera```.

	You can change other models. Then, let's change the other model, ResNet-18!!

	```shell
	$ python3 imagenet.py --network=resnet-18 /dev/video0
	```
	
	There is a table shows jetson-inference repository supports.
	
| Network       | CLI argument   | NetworkType enum |
|---------------|----------------|------------------|
| AlexNet       | `alexnet`      | `ALEXNET`        |
| GoogleNet     | `googlenet`    | `GOOGLENET`      |
| GoogleNet-12  | `googlenet-12` | `GOOGLENET_12`   |
| ResNet-18     | `resnet-18`    | `RESNET_18`      |
| ResNet-50     | `resnet-50`    | `RESNET_50`      |
| ResNet-101    | `resnet-101`   | `RESNET_101`     |
| ResNet-152    | `resnet-152`   | `RESNET_152`     |
| VGG-16        | `vgg-16`       | `VGG-16`         |
| VGG-19        | `vgg-19`       | `VGG-19`         |
| Inception-v4  | `inception-v4` | `INCEPTION_V4`   |


	You could see the various models before build makes. In details, please visit [this github site](https://github.com/dusty-nv/jetson-inference).

2. Object Detection 
	
	Object Detection models show the bounded boxes we trained. 
	
	```shell
	$ python3 detectnet.py /dev/video0
	```
	
	In this case, we can check the default network is SSD-MobileNet-V2. and the below table is network model list. and [COCO](https://cocodataset.org/#home) is the most used Image Datasets. There are many labelings(Person, chair, animals, ... and so on) in COCO Datasets. For training them, we need much times and expensive devices. So we simply introduce object detection's inference.

	| Network                 | CLI argument       | NetworkType enum   | Object classes       |
	| ------------------------|--------------------|--------------------|----------------------|
	| SSD-Mobilenet-v1        | `ssd-mobilenet-v1` | `SSD_MOBILENET_V1` | 91 CoCo classes	   |
	| SSD-Mobilenet-v2        | `ssd-mobilenet-v2` | `SSD_MOBILENET_V2` | 91 CoCo classes      |
	| SSD-Inception-v2        | `ssd-inception-v2` | `SSD_INCEPTION_V2` | 91 CoCo classes      |
	| DetectNet-COCO-Dog      | `coco-dog`         | `COCO_DOG`         | dogs                 |
	| DetectNet-COCO-Bottle   | `coco-bottle`      | `COCO_BOTTLE`      | bottles              |
	| DetectNet-COCO-Chair    | `coco-chair`       | `COCO_CHAIR`       | chairs               |
	| DetectNet-COCO-Airplane | `coco-airplane`    | `COCO_AIRPLANE`    | airplanes            |
	| ped-100                 | `pednet`           | `PEDNET`           | pedestrians          |
	| multiped-500            | `multiped`         | `PEDNET_MULTI`     | pedestrians, luggage |
	| facenet-120             | `facenet`          | `FACENET`          | faces                |

3. Semantic Segmentation.

	Object 

## Introduction : NVIDIA
