## Install : Prerequisies and Dependencies
First of all, we need to install "Nvidia Jetson nano toolkit" following the [official instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write). 

```
It might take 7 mins ~ 15 mins.
(in the case that you downloaded Ether Program and jetson image, already)
Note that we don't care whatever your OS(e.g. Windows, Linux, Mac) is.
You need just a desktop which can download faster.
Also, you can set your jetson ID yourself. (I set SierrabaseCourse.)
```

Before install tensorflow, you have to check Jetpack version (**It is so important!**). The way to check jetpack version is very simple. If your download zip name is ```jetson-nano-jp451-sd-card-image.zip```, this means you've installed JetPack 4.5.1 version.

In my case, these packages are set up

- JetPack 4.5.1
- L4T 32.5.1
- TensorRT 7.1.3
- CUDA 10.2
- CuDNN 8.0
- OpenCV 4.1.1
- VPI 1.0

by [this link](https://developer.nvidia.com/embedded/jetpack). If you want to know older version, [check this site](https://developer.nvidia.com/embedded/jetpack-archive)

Also, the suitable tensorflow version can be found in [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel). My tensorflow version is 2.4.0!! 

Now, let's begin installation packages for Jetson nano kit. You must proceed with Jetson nano. Jetson nano OS is based on ```Ubuntu```. So, it may be not diffcult. Now, let's open Terminal with commands ('Ctrl' + 'Alt' + 'T').

1. Install System Packages
    ```shell
    $ sudo apt-get update
    $ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran tree
   ```

2. Upgrade pip3
    ```shell
    $ sudo apt-get install python3-pip
    $ sudo pip3 install -U pip testresources setuptools==49.6.0
    ```

3. Install Python Package Dependencies
    ```shell
    $ sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
    ```

## Install : tensorflow

Many Developers still find tensorflow(tf) version under 2. You must choose either 1 or 2 ```(Recommend to follow phase 1, not 2)```

1. Install latest version tf > 2.x (Recommend in this course!)
    ```shell
    $ sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow
    ```

2. Or, you can also install lower version tf < 2.x (If you've finished step 1, never do this!)
    ```shell
    $ sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 ‘tensorflow<2’
    ```

## Download : Project

Congraturations!! We start projects

I set ```/home/{your_nano_id}/AI_beginner_course```

1. Go into your home directory
	```shell
	$ cd
	```
2. Download project from github
	```shell
	$ git clone https://github.com/SierrabaseLab/AI_beginner_course.git
	```

3. Prepare next folder to ```./DL_course/Image_classification```
	```shell
	$ cd AI_beginner_course/DL_course/Image_Classification/
	```

The next "MNIST Tutorial" Section continues...
