# Requirements

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


# Convert jpg_images to png_images and clean ICC profile from PNGs in png_images and png_masks
```python3 convert_to_png_and_clean.py```

# Convert png_masks to grayscale images with intensities representing classes in folder png_masks_8bit as 8-bit depth, with values 0 - 255 (or 16-bit depth in folder png_masks_16bit, if needed for more that 256 classes - including background class 0)
```python3 convert_masks_to_grayscale.py```

# Run image segmentation
```python3 segmentation.py```
