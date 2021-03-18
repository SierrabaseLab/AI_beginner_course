## Install : Prerequisies and Dependencies
First of all, we need to install "Nvidia Jetson nano toolkit" following the [official instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)

If Installation has been completed, then run jetson nano toolkit. We only proceed with Jetson Nano.
Before install tensorflow, you have to check Jetpack version(**It is so important!**) The way to check jetpack version is very simple. 

Your downloaded zip file will be ```jetson-nano-jp451-sd-card-image.zip```. This means you've installed JetPack 4.5.1 version.
In my case, these packages are set up

- JetPack 4.5.1
- L4T 32.5.1
- TensorRT 7.1.3
- CUDA 10.2
- CuDNN 8.0
- OpenCV 4.1.1
- VPI 1.0

by [this link](https://developer.nvidia.com/embedded/jetpack).

Also, the suitable tensorflow version can be found in [here](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.html#tf-jetson-rel). My tensorflow version is 2.4.0!!

1. Install System Packages
    ```shell
    sudo apt-get update
    sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    ```

2. Upgrade pip3
    ```shell
    sudo apt-get install python3-pip
    sudo pip3 install -U pip testresources setuptools==49.6.0
    ```

3. Install Python Package Dependencies
    ```shell
    sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
    ```

## Install : tensorflow

1. Install latest version tf > 2.x (Recommend! As I told you, you must check JetPack version and tensorflow version)
    ```shell
    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow
    ```

2. Install lower version tf < 2.x
    ```shell
    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 ‘tensorflow<2’
    ```
