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

import OpenEXR, array

# Triangular hat weighting function
w = np.arange(256)
w[w>127] = 255.-w[w>127]

# Set epsilon
eps = 1e-8

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
      A[k, i+256] = -w[z]
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

if len(sys.argv)!=3 and len(sys.argv)!=4:
  print('    Usage: python hdr.py <image list file> <number of points> [hdr image]')
  sys.exit(-1)
else:
  inFile = sys.argv[1]
  nPoints = int(sys.argv[2])

  if len(sys.argv)==4:
    outImage = sys.argv[3]

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
plt.xlabel('Intensity')
plt.ylabel('Response function')

plt.savefig('z-lnE.png')
plt.cla()

# Generate the radiance map
print('Generate the radiance map...')
radm = np.zeros((imgHeight, imgWidth, 3))

for r in xrange(imgHeight):
  for c in xrange(imgWidth):
    zr = imgs[:, r, c, 2]
    zg = imgs[:, r, c, 1]
    zb = imgs[:, r, c, 0]
    radm[r, c, 2] = np.exp(np.sum(w[zr]*(gr[zr]-lnts))/(np.sum(w[zr])+eps))
    radm[r, c, 1] = np.exp(np.sum(w[zg]*(gg[zg]-lnts))/(np.sum(w[zg])+eps))
    radm[r, c, 0] = np.exp(np.sum(w[zb]*(gb[zb]-lnts))/(np.sum(w[zb])+eps))

# Output the EXR image
exr = OpenEXR.OutputFile(outImage+'.exr', OpenEXR.Header(*radm.shape[:2][::-1]))

r = array.array('f', radm[:, :, 2].reshape(-1)).tostring()
g = array.array('f', radm[:, :, 1].reshape(-1)).tostring()
b = array.array('f', radm[:, :, 0].reshape(-1)).tostring()

exr.writePixels({'R': r, 'G': g, 'B': b})

# Output radiance map
rad = np.log(radm)

rad[:, :, 0] = (rad[:, :, 0]-rad[:, :, 0].min())/(rad[:, :, 0].max()-rad[:, :, 0].min()+eps)*255
rad[:, :, 1] = (rad[:, :, 1]-rad[:, :, 1].min())/(rad[:, :, 1].max()-rad[:, :, 1].min()+eps)*255
rad[:, :, 2] = (rad[:, :, 2]-rad[:, :, 2].min())/(rad[:, :, 2].max()-rad[:, :, 2].min()+eps)*255

rad = cv2.applyColorMap(rad.astype(np.uint8), cv2.COLORMAP_JET)
cv2.imwrite('rad.png', rad)

# Show the draw points
for pt in pts:
  cv2.circle(imgs[0], pt[::-1], 3, (0, 0, 255))

cv2.imwrite('pts.jpg', imgs[0])

print('HDR done!')
