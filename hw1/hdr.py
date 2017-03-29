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
w = np.arange(256)
w[w>127] = 255.-w[w>127]

# Fill the Sparse linear system
def getOptimizeMatrix(imgs, lnts, pts, lamb):
  nImgs = imgs.shape[0]
  nPts = len(pts)

  # Sparse matrix
  A = np.zeros((nImgs*nPts+255, 256+nPts), np.float32)
  b = np.zeros((nImgs*nPts+255, 1), np.float32)

  k = 0
  for j, img in enumerate(imgs):
    for i, pt in enumerate(pts):
      z = img[pt]
      A[k, z] = w[z]
      A[k, i+256] = w[z]
      b[k] = lnts[j]*w[z]
      k += 1
  A[k, 127] = 1.
  k += 1

  for i in xrange(254):
    A[k, i] = w[i+1]*lamb
    A[k, i+1] = -2.*w[i+1]*lamb
    A[k, i+2] = w[i+1]*lamb
    k+=1

  return A, b

# Calculate the respondse function
def getRespondseFunction(A, b):
  invA = np.linalg.pinv(A)
  x = np.dot(invA, b)
  g = x[:256].reshape(-1)
  le = x[256:].reshape(-1)

  return g, le

# Default parameters
outImage = 'img'
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
lnts = np.log(expts)

# Generate random points
imgHeight, imgWidth, imgChannels = imgs[0].shape

pts = [(np.random.randint(0, imgHeight), np.random.randint(0, imgWidth))
       for _ in xrange(nPoints)]

# Fill the optimization matrix
lamb = 1.
Ar, br = getOptimizeMatrix(imgs[:, :, :, 2].reshape(-1, imgHeight, imgWidth), lnts, pts, lamb)
Ag, bg = getOptimizeMatrix(imgs[:, :, :, 1].reshape(-1, imgHeight, imgWidth), lnts, pts, lamb)
Ab, bb = getOptimizeMatrix(imgs[:, :, :, 0].reshape(-1, imgHeight, imgWidth), lnts, pts, lamb)

# Solve the optimization eqation by SVD
gr, ler = getRespondseFunction(Ar, br)
gg, leg = getRespondseFunction(Ag, bg)
gb, leb = getRespondseFunction(Ab, bb)

import matplotlib.pyplot as plt

plt.plot(np.arange(len(gr)), gr, 'r')
plt.plot(np.arange(len(gg)), gg, 'g')
plt.plot(np.arange(len(gb)), gb, 'b')
plt.legend(['R', 'G', 'B'])

plt.savefig('z-lnE.png')

# Generate the radiance map
print('Generate the radiance map...')
radm = np.zeros((imgHeight, imgWidth, 3))

for r in xrange(imgHeight):
  for c in xrange(imgWidth):
    zr = imgs[:, r, c, 2]
    zg = imgs[:, r, c, 1]
    zb = imgs[:, r, c, 0]
    radm[r, c, 2] = np.exp(np.sum(w[zr]*(gr[zr]-lnts))/np.sum(w[zr]))
    radm[r, c, 1] = np.exp(np.sum(w[zg]*(gg[zg]-lnts))/np.sum(w[zg]))
    radm[r, c, 0] = np.exp(np.sum(w[zb]*(gb[zb]-lnts))/np.sum(w[zb]))

# Output the HDR image
cv2.imwrite(outImage+'.hdr', radm)

radm = np.log(radm)

radm[:, :, 0] = (radm[:, :, 0]-radm[:, :, 0].min())/(radm[:, :, 0].max()-radm[:, :, 0].min())*255
radm[:, :, 1] = (radm[:, :, 1]-radm[:, :, 1].min())/(radm[:, :, 1].max()-radm[:, :, 1].min())*255
radm[:, :, 2] = (radm[:, :, 2]-radm[:, :, 2].min())/(radm[:, :, 2].max()-radm[:, :, 2].min())*255

radm = cv2.applyColorMap(radm.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite('rad.png', radm)

for pt in pts:
  cv2.circle(imgs[0], pt[::-1], 3, (0, 0, 255))

cv2.imwrite('pts.jpg', imgs[0])

print('HDR done!')
