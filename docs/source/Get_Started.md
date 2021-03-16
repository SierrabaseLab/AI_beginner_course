## Install : Prerequisies and Dependencies

1. Install "Nvidia Jetson nano toolkit" following the [official instructions](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)

Then, you have to check tensorflow version [in this link](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform-release-notes/tf-jetson-rel.ht4.5ml)

In my case, these packages are set up

- JetPack 4.5.1
- L4T 32.5.1
- TensorRT 7.1.3
- CUDA 10.2
- CuDNN 8.0
- OpenCV 4.1.1
- VPI 1.0

Therefore, my tensorflow version is 2.4.0

2. Install Sysem Packages
    ```shell
    sudo apt-get update
    sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
    ```

3. Upgrade pip3
    ```shell
    sudo apt-get install python3-pip
    sudo pip3 install -U pip testresources setuptools==49.6.0
    ```

4. Install Python Package Dependencies
    ```shell
    sudo pip3 install -U numpy==1.19.4 future==0.18.2 mock==3.0.5 h5py==2.10.0 keras_preprocessing==1.1.1 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
    ```

## Install : tensorflow

1. Install latest version tf > 2.x (Recommend)
    ```shell
    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow
    ```

2. Install lower version tf < 2.x
    ```shell
    sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 ‘tensorflow<2’
    ```
