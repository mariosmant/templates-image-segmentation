# Requirements

## CUDA Toolkit
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
nano ~/.bashrc
```
add the below lines:
```
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
export PATH=/usr/local/cuda-11.8/bin:$PATH
```
```
source ~/.bashrc
sudo nano /etc/ld.so.conf
```
add the below line:
```
/usr/local/cuda-11.8/lib64
```
Ctrl+O<br />
Enter<br />
Ctrl+X
```
sudo ldconfig
```

## CUDNN
```
cd ~/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
sudo cp include/cudnn*.h /usr/local/cuda-11.8/include
sudo cp lib/libcudnn* /usr/local/cuda-11.8/lib64
sudo chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
cd ~
gcc -o test_cudnn test_cudnn.c -I/usr/local/cuda-11.8/include -L/usr/local/cuda-11.8/lib64 -lcudnn
./test_cudnn
```

## TensorRT
```
cd ~
tar -xvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz TensorRT-8.6.1.6
sudo mv TensorRT-8.6.1.6 /usr/local/TensorRT-8.6.1.6
```

## Conda
```
conda create --name tfsegmentation python=3.10.16
conda activate tfsegmentation
```

## Install Tensorflow
```
conda activate tfsegmentation
python3 -m pip install tensorflow==2.13.1
```

## Python Versions
Python 3.10.16

python3 -m pip install tensorflow==2.13.1
python3 -m pip install opencv-python==4.10.0.84
python3 -m pip install matplotlib==3.10.0
python3 -m pip install tqdm==4.67.1
python3 -m pip install scikit-learn==1.6.0 scipy==1.14.1
python3 -m pip install git+https://github.com/tensorflow/examples.git@652ee34ff046946c36b8aed5d97ecebab0699f7e

python3 -m pip show numpy
Should show 1.24.3
If not: python3 -m pip install numpy==1.24.3

### Use requirements.txt
python3 -m pip install -r requirements.txt

## Folder structures and datasets
- Add in `jpg_images` the train/test datasets' images (.jpg).
- Add in `png_masks` the images with masks corresponding to images in jpg_images (should have the same filename with jpg_images, except extension is .png).

# Preprocess Images
## Combined
```python3 preprocess_images.py```

## Manually
### Convert jpg_images to png_images and clean ICC profile from PNGs in png_images and png_masks
```python3 convert_to_png_and_clean.py```

### Convert png_masks to grayscale images with intensities representing classes in folder png_masks_8bit as 8-bit depth, with values 0 - 255 (or 16-bit depth in folder png_masks_16bit, if needed for more that 256 classes - including background class 0)
```python3 convert_masks_to_grayscale.py```

# Run image segmentation
```python3 segmentation.py```

# TensorBoard
```tensorboard --logdir=model/custom-model/logs```
