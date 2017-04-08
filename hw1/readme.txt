# README for Digital Visual Effects homework 1
* Author: Cheng-Shih Wong, r04945028@ntu.edu.tw
* Author: Hong-Yi Chen, r04922099@ntu.edu.tw

## Environment
* Operating System: **Ubuntu 16.04**
* Central Process Unit: Intel(R) Xeon(R) CPU E3-1231 v3 @ 3.40GHz
* Random Access Memory: 16 GB
* Programming Language: **Python 2.7**
* Libraries:
  * **OpenCV3** -- image IO, bilateral filter
  * **numpy** -- matrix operation
  * **OpenEXR** -- .exr image output
  * Tone mapping software: [**Luminance HDR**](https://github.com/LuminanceHDR)

## Usage
1. `python src/genImageList.py <img_dir> [img_list_file]` -- generate `image_list_file`
2. `python src/hdr.py <img_list_file> <number_of_points> [HDR_img]` -- reconstruct HDR image as `HDR_img.exr`
3. `python src/tonemap.py <HDR_img.exr> <output_image>` -- tone mapping by the `HDR_img.exr` to develop radiance map

---

## genImageList.py
Generate file of image list for hdr.py

* for digital visual effects homework1 -- `hdr.py`
* generate a file with <image-location> and <exposure-time>
  splits by space per line

---

## hdr.py
Calculate HDR image with given image list file

* for digital visual effects homework1
* generate a hdr image by <image-list>

---

## tonemap.py
HDR image tone mapping

* for digital visual effects homework1
* generate a tone mapping image by an exr HDR image

