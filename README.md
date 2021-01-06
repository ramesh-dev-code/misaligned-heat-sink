# Image Classification of Misaligned Heat Sink
### Objective
To design the CNN-based deep neural network (DNN) to infer the presence of misaligned heat sink in the device under test   
### Heat Sink Images   
**PASS: Algined Heat Sink**   
![Optional Text](images/pass.jpg)   
**FAIL: Misalgined Heat Sink**   
![Optional Text](images/fail.jpg)   

### Architecture of VGG16 Model      
![](https://i.imgur.com/fshCcha.png)   
### Proposed Model   
![](https://i.imgur.com/kDrjwRs.png)
### Data Acquisition   
1. Used the python script capture_image.py to capture the webcam images   
2. Labeled the two output classes to be classifed by the proposed model as: 0-FAIL (Misaligned or no heat sink), 1-PASS (Correctly aligned heat sink)
3. Captured the webcam images with the following experimental setup
#### Experimental Setup
![](https://i.imgur.com/9Hw9XEL.png)   
4. Collected 2250 images for FAIL and 2054 images for PASS

## Training Platform Setup
### Training Platform
OS: Ubuntu 16.04
Kernel: 4.16.0-041600-generic   
Memory: 32 GB   
CPU: Intel® Core™ i7-8700K CPU @ 3.70GHz × 12   
GPU: GeForce GTX 1080 x 2   

### Prerequisites
1. Remove the previous installations of NVIDIA driver, CUDA toolkit and cuDNN libraries
```
sudo apt purge --remove '^nvidia'
sudo apt purge --remove '^cuda'
sudo apt-get clean && apt-get autoremove
dpkg --configure -a
rm -rf /usr/local/cuda*
sudo apt-get upgrade
sudo apt-get update
```
2. Check the [dependenices to install Tensorflow 2.4.0](https://www.tensorflow.org/install/gpu) with GPU support
3. Install Python 3.8
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-dev python3.8-venv
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
sudo update-alternatives --config python3
python3 -m pip install --upgrade pip
pip3 install -U virtualenv
```
4. [Update the gcc and g++](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5) modules to the latest version
```
sudo apt-get install -y software-properties-common python-software-properties
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install g++-7 -y
sudo update-alternatives \
    --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
    --slave /usr/bin/gcc-ar gcc-ar /usr/bin/gcc-ar-7 \
    --slave /usr/bin/gcc-nm gcc-nm /usr/bin/gcc-nm-7 \
    --slave /usr/bin/gcc-ranlib gcc-ranlib /usr/bin/gcc-ranlib-7 \   
    --slave /usr/bin/g++ g++ /usr/bin/g++-7
```
Update the sym link gcc in /etc/alternatives
```
rm /etc/alternatives/gcc
sudo ln -s /usr/bin/gcc-7 /etc/alternatives/gcc
sudo ln -s /usr/bin/g++-7 /etc/alternatives/g++
```
Verify the installed gcc version
![](https://i.imgur.com/js6ekSE.png)   
### Installation of CUDA Toolkit and NVIDIA Driver

**Note**: The NVIDIA driver is installed as part of the CUDA Toolkit. There is no need of prior installation of the NVIDIA driver. If installed, remove the previous version

#### [Installation Steps](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal)

1. Open a new console by pressing Ctrl+Alt+F1. Stop the display manager service to stop the X server
```
sudo service lightdm stop
```
2. Set the run level to VGA
```
sudo init 3
```
3. Set the environment variable CC to point to the gcc path
```
export CC=/usr/bin/gcc-7
```
4. Install CUDA toolkit and NVIDIA driver
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda-repo-ubuntu1604-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604-11-0-local_11.0.2-450.51.05-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu1604-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```
Add the installation path of CUDA toolkit to the PATH variable in ~/.bashrc 
```
export PATH=$PATH:/usr/local/cuda/bin/
```
Update the bashrc file
```
source ~/.bashrc
```

5. Reboot the machine to verify the installation of driver and CUDA

**Verification of CUDA 11.0 Toolkit installation**   
![](https://i.imgur.com/wesT7iZ.png)   
**Verification of NVIDIA driver installation**
![](https://i.imgur.com/l1ckdBE.png)

**Verification of the interaction between CUDA toolkit and GPUs**

Build the CUDA samples after configuring the g++ compiler
```
cd /usr/local/cuda/samples/
make
```
Execute the CUDA application 
```
./bin/x86_64/linux/release/deviceQuery
```
![](https://i.imgur.com/5tdkWno.png)
![](https://i.imgur.com/S92hlT6.png)

### [Installation of cuDNN Libraries](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

1. [Download the debian files of the cuDNN libraries](https://developer.nvidia.com/cudnn-download-survey) 

2. Install the runtime library
```
dpkg -i libcudnn8_8.0.5.39-1+cuda11.0_amd64.deb
```

3. Install the developer library
```
dpkg -i libcudnn8-dev_8.0.5.39-1+cuda11.0_amd64.deb
```

4. Install the code samples and the cuDNN library documentation
```
dpkg -i libcudnn8-samples_8.0.5.39-1+Installation of Tensorflow 2.4.0cuda11.0_amd64.deb
```

5. Verification of cuDNN application
```
cd /home/test/Documents/DeepLearning/Classification/AOI/cuDNN
cp -r /usr/src/cudnn_samples_v8/ ./
make clean && make
./mnistCUDNN
```
**Result**
![](https://i.imgur.com/6yKgRrq.png)
![](https://i.imgur.com/y7kIjtU.png)

### Installation of Tensorflow 2.4.0

1. Create the Python virutal environment
```
cd /home/test/Documents/DeepLearning
mkdir venv_tf
python3 -m venv ./venv_tf/
source ./venv_tf/bin/activate
```
2. Upgrade the pip version
python3 -m pip install --upgrade pip

3. Install Tensorflow
pip3 install tensorflow

4. Update the path of python.exe in the virtual environment to the Eclipse IDE

### Errors and Workaround

#### Error
```
dpkg: error processing archive /var/cache/apt/archives/cuda-repo-ubuntu1604-10-0-local-10.0.130-410.48_1.0-1_amd64.deb
```
#### Workaround
```
sudo dpkg -i --force-overwrite /var/cache/apt/archives/cuda*
```
