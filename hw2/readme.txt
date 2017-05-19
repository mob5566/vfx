# README for Digital Visual Effects project 2
* Author: Cheng-Shih Wong, r04945028@ntu.edu.tw
* Author: Hong-Yi Chen, r04922099@ntu.edu.tw

## Environment
* Operating System: **Ubuntu 16.04**
* Central Process Unit: Intel(R) Xeon(R) CPU E3-1231 v3 @ 3.40GHz
* Random Access Memory: 16 GB
* Programming Language: **Python 2.7**
* Libraries:
  * **OpenCV 3** -- image IO, bilateral filter
  * **numpy** -- matrix operation
  * **scipy.spatial.KDTree** -- *k*-dimensional tree

## Usage
1. The input txt file should contain lines with `<image_location> <image_focal_length>`
2. `python src/panorama.py input_image/input_image.txt result/` -- generate panorama images

---

## genImageList.py
Generate file of image list for desinated directory

* generate a file with <image-location> per line from an directory

---

## msop.py
Calcuate the Multi-Scale Oriented Patch features of an image

---

## panorama.py
Generate all panaramas from a given images file

