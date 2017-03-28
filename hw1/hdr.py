'''
# hdr.py -- Calculate HDR image with given image list file

* for digital visual effects homework1
* generate a hdr image by <image-list>

'''

from __future__ import print_function

import os
import sys

import cv2
import numpy as np

# Default parameters
outImage = 'img.hdr'
nPoints = 50

if len(sys.argv)!=2 and len(sys.argv)!=3:
  print('    Usage: python hdr.py <image list file> [hdr image]')
  sys.exit(-1)
else:
  inFile = sys.argv[1]

  if len(sys.argv)==3:
    outImage = sys.argv[2]

# Get the name of images and its exposure time
imgs = []

with open(inFile, 'rb') as f:
  for line in f.readlines():
    fn, exptime = line.split()
    exptime = float(exptime)

    imgs.append((fn, exptime))

# Get an image for demonstration
img = cv2.imread(imgs[0][0])

# Generate random points
imgHeight, imgWidth, imgChannels = img.shape

pts = [(np.random.randint(0, imgWidth), np.random.randint(0, imgHeight))
       for _ in xrange(nPoints)]

for pt in pts:
  cv2.circle(img, tuple(pt), 3, (0, 0, 255))

