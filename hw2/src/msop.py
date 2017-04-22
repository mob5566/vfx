'''
# msop.py -- Multi-Scale Oriented Patches feature

* for digital visual effects homework1
* generate a hdr image by <image-list>

'''

from __future__ import print_function

import os
import sys

import cv2
import numpy as np

from itertools import product

# Epsilon
eps = 1e-8

class MSOP():
  def __init__(self, numFeat=500, pyrLevel=5, fhmt=10.0):
    self.nf = numFeat       # Number of features
    self.pyrl = pyrLevel    # Number of levels of pyramid
    self.ft = fhmt          # Harmonic mean threshold
    self.kp = None          # Keypoints
  
  def detect(self, img):

    # Input array should be a numpy array (image)
    assert(type(img)==np.ndarray)
    
    # Obtain gray scale image
    if len(img.shape)==3:
      gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
      gimg = img.copy()

    # Build pyramid
    h, w = gimg.shape
    p = gimg.astype(float)
    self.kp = []

    for l in xrange(self.pyrl):
      pdx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=5)
      pdy = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=5)

      pgdx = cv2.GaussianBlur(pdx, (5, 5), 1.0)
      pgdy = cv2.GaussianBlur(pdy, (5, 5), 1.0)

      ch, cw = pgdx.shape

      fhm = np.zeros((ch, cw))

      for x, y in product(range(cw), range(ch)):
        gp = np.array([[pgdx[y, x]], [pgdy[y, x]]])

        # Harris matrix
        hl = cv2.GaussianBlur(np.dot(gp, gp.T), (5, 5), 1.5)

        # Harmonic mean
        fhm[y, x] = (hl[0, 0]*hl[1, 1])/(hl[0, 0]+hl[1, 1]+eps)

      # Detect key points
      for x, y in product(range(1, cw-1), range(1, ch-1)):
        if fhm[y, x]>=self.ft and np.all(fhm[y, x]>=fhm[y-1:y+2, x-1:x+2].reshape(-1)):
          self.kp.append((x, y, l, fhm[y, x]))

      p = cv2.GaussianBlur(p, (5, 5), 1.0)
      p = cv2.resize(p, (p.shape[1]/2, p.shape[0]/2))

    return self.kp
  
  # Adaptive Non-Maximal Suppression
  def anms(self):
    if not self.kp:
      return None
    
    # Robust constant
    cr = 0.9

    kp = self.kp

    n = len(kp)
    newkp = []
    dis = lambda i, j: (kp[i][0]-kp[j][0])**2+(kp[i][1]-kp[j][1])**2

    for i in xrange(n):
      minr = 1e8

      for j in xrange(n):
        if j==i: continue

        if kp[i][3] < cr*kp[j][3] and dis(i, j)<minr:
          minr = dis(i, j)

      newkp.append((kp[i][0], kp[i][1], kp[i][2], minr))

    newkp.sort(key=lambda x: x[3], reverse=True)

    return newkp[:self.nf]

def main():
  img = cv2.imread('data/parrington/prtn00.jpg')

  fd = MSOP()
  kp = fd.detect(img)
  nkp = fd.anms()

  # Show detected key points
  for p in nkp:
    cv2.circle(img, (p[0]*2**p[2], p[1]*2**p[2]), 2*2**p[2], (0, 0, 255))

  cv2.imshow('tmp', img)
  cv2.waitKey()
  cv2.destroyAllWindows()

  return kp, nkp

if __name__=='__main__':
  kp, nkp = main()
  
