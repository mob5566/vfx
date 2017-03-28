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

# Triangular hat weighting function
w = lambda(z): float(z) if z<=127 else float(255-z)

# Fill the Sparse linear system
def getOptimizeMatrix(imgs, expts, pts, lamb):
  nImgs = imgs.shape[0]
  nPts = len(pts)

  # Sparse matrix
  A = np.zeros((nImgs*nPts+255, 256+nPts), np.float32)
  b = np.zeros((nImgs*nPts+255, 1), np.float32)

  k = 0
  for j, img in enumerate(imgs):
    for i, pt in enumerate(pts):
      z = img[pt]
      A[k, z] = w(z)
      A[k, i+256] = w(z)
      b[k] = np.log(expts[j])*w(z)
      k += 1
  A[k, 127] = 1.
  k += 1

  for i in xrange(254):
    A[k, i] = w(i+1)*lamb
    A[k, i+1] = -2.*w(i+1)*lamb
    A[k, i+2] = w(i+1)*lamb
    k+=1

  return A, b

# Calculate the respondse function
def getRespondseFunction(A, b):
  invA = np.linalg.pinv(A)
  x = np.dot(invA, b)
  g = x[:256]
  le = x[256:]

  return g, le

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
expts = []

with open(inFile, 'rb') as f:
  for line in f.readlines():
    fn, exptime = line.split()
    exptime = float(exptime)

    imgs.append(cv2.imread(fn))
    expts.append(exptime)

imgs = np.array(imgs)

# Generate random points
imgHeight, imgWidth, imgChannels = imgs[0].shape

pts = [(np.random.randint(0, imgHeight), np.random.randint(0, imgWidth))
       for _ in xrange(nPoints)]

# Fill the optimization matrix
Ar, br = getOptimizeMatrix(imgs[:, :, :, 0].reshape(-1, imgHeight, imgWidth), expts, pts, 10.)
Ag, bg = getOptimizeMatrix(imgs[:, :, :, 1].reshape(-1, imgHeight, imgWidth), expts, pts, 10.)
Ab, bb = getOptimizeMatrix(imgs[:, :, :, 2].reshape(-1, imgHeight, imgWidth), expts, pts, 10.)

# Solve the optimization eqation by SVD
gr, ler = getRespondseFunction(Ar, br)
gg, leg = getRespondseFunction(Ag, bg)
gb, leb = getRespondseFunction(Ab, bb)


