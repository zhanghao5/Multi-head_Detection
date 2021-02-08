# Multi-head Object Detection Architecure

## (neural networks for object detection)

Our code is based on https://github.com/AlexeyAB/darknet/wiki
and developed with C and CUDA

### Requirements

* Windows or Linux
* **CMake >= 3.12**: https://cmake.org/download/
* **CUDA >= 10.0**: https://developer.nvidia.com/cuda-toolkit-archive (on Linux do [Post-installation Actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions))
* **OpenCV >= 2.4**: use your preferred package manager (brew, apt), build from source using [vcpkg](https://github.com/Microsoft/vcpkg) or download from [OpenCV official site](https://opencv.org/releases.html) (on Windows set system variable `OpenCV_DIR` = `C:\opencv\build` - where are the `include` and `x64` folders [image](https://user-images.githubusercontent.com/4096485/53249516-5130f480-36c9-11e9-8238-a6e82e48c6f2.png))
* **cuDNN >= 7.0** https://developer.nvidia.com/rdp/cudnn-archive (on **Linux** copy `cudnn.h`,`libcudnn.so`... as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installlinux-tar , on **Windows** copy `cudnn.h`,`cudnn64_7.dll`, `cudnn64_7.lib` as desribed here https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows )
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* on Linux **GCC or Clang**, on Windows **MSVC 2017/2019** https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community


### How to compile on Linux (using `make`)

Just do `make` in the darknet directory. (You can try to compile and run it on Google Colab in cloud [link](https://colab.research.google.com/drive/12QusaaRj_lUwCGDvQNfICpa7kA7_a2dE) (press «Open in Playground» button at the top-left corner) and watch the video [link](https://www.youtube.com/watch?v=mKAEGSxwOAY) )
Before make, you can set such options in the `Makefile`: [link](https://github.com/AlexeyAB/darknet/blob/9c1b9a2cf6363546c152251be578a21f3c3caec6/Makefile#L1)

* `GPU=1` to build with CUDA to accelerate by using GPU (CUDA should be in `/usr/local/cuda`)
* `CUDNN=1` to build with cuDNN v5-v7 to accelerate training by using GPU (cuDNN should be in `/usr/local/cudnn`)
* `CUDNN_HALF=1` to build for Tensor Cores (on Titan V / Tesla V100 / DGX-2 and later) speedup Detection 3x, Training 2x
* `OPENCV=1` to build with OpenCV 4.x/3.x/2.4.x - allows to detect on video files and video streams from network cameras or web-cams
* `DEBUG=1` to bould debug version of Yolo
* `OPENMP=1` to build with OpenMP support to accelerate Yolo by using multi-core CPU
* `LIBSO=1` to build a library `darknet.so` and binary runable file `uselib` that uses this library. Or you can try to run so `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib test.mp4` How to use this SO-library from your own code - you can look at C++ example: https://github.com/AlexeyAB/darknet/blob/master/src/yolo_console_dll.cpp
    or use in such a way: `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights test.mp4`
* `ZED_CAMERA=1` to build a library with ZED-3D-camera support (should be ZED SDK installed), then run
    `LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH ./uselib data/coco.names cfg/yolov4.cfg yolov4.weights zed_camera`

To run Darknet on Linux use examples from this article, just use `./darknet` instead of `darknet.exe`, i.e. use this command: `./darknet detector test ./cfg/coco.data ./cfg/yolov4.cfg ./yolov4.weights`

### How to compile on Windows (using `CMake`)

This is the recommended approach to build Darknet on Windows.

1. Install Visual Studio 2017 or 2019. In case you need to download it, please go here: [Visual Studio Community](http://visualstudio.com)

2. Install CUDA (at least v10.0) enabling VS Integration during installation.

3. Open Powershell (Start -> All programs -> Windows Powershell) and type these commands:

```PowerShell
PS Code\>              git clone https://github.com/microsoft/vcpkg
PS Code\>              cd vcpkg
PS Code\vcpkg>         $env:VCPKG_ROOT=$PWD
PS Code\vcpkg>         .\bootstrap-vcpkg.bat
PS Code\vcpkg>         .\vcpkg install darknet[full]:x64-windows #replace with darknet[opencv-base,cuda,cudnn]:x64-windows for a quicker install of dependencies
PS Code\vcpkg>         cd ..
PS Code\>              git clone https://github.com/AlexeyAB/darknet
PS Code\>              cd darknet
PS Code\darknet>       .\build.ps1
```

### How to run the program to reproduce experiment
Our network configuration files, trained weights files, and raw data are 
store in Google drive at [link](https://drive.google.com/file/d/1jvZAZkdWORh1FrQTzpAH3GPo7Dg0F2o2/view?usp=sharing)

for the data configure file (.data format)
```ini
# nubmer of classes to be trained
classes= 13
# base training image list file
train  = data/train.txt
# continual training image list file
# swith to use when continual training
# train = data/new.txt
# testing image list file
valid  = data/test.txt
# object name file 
names = cfg/obj-auto.names
# intermediate and final weights store path
backup = backup/
```

#### Base train (train from scratch)
`./darknet detector train auto.data base_pre.cfg -dont_show`

#### Continual learning
`./darknet detector train auto_new.data small13.cfg base_pre_final.weights`

#### Testing
`./darknet detector map auto.data 1head13c.cfg base_pre_final.weights small13_final.weights -points 101 -n_heads 1 -gate -dont_show`
where -n_heads option means how many incremental heads would be contribute for final prediction, -gate option means using gate function