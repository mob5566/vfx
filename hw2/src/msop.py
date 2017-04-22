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

class MSOP(object):
  def __init__(self, numFeat=500, pyrLevel=3, fhmt=10.0):
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
    self.pyramid = [p.copy()]

    for l in xrange(1, self.pyrl):
      p = cv2.GaussianBlur(p, (5, 5), 1.0)
      p = cv2.resize(p, (p.shape[1]/2, p.shape[0]/2))
      self.pyramid.append(p.copy())

    for l in xrange(self.pyrl):
      p = self.pyramid[l]
      pdx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=5)
      pdy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=5)

      pgdx = cv2.GaussianBlur(pdx, (5, 5), 1.0)
      pgdy = cv2.GaussianBlur(pdy, (5, 5), 1.0)

      pgodx = cv2.GaussianBlur(pdx, (5, 5), 4.5)
      pgody = cv2.GaussianBlur(pdy, (5, 5), 4.5)

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
        nei = fhm[y-1:y+2, x-1:x+2]
        if fhm[y, x]>=self.ft and np.all(fhm[y, x]>=nei.reshape(-1)):
          sx, sy = MSOP.spa((x, y), nei)
          isx = int(round(sx))
          isy = int(round(sy))
          dx = pgodx[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)
          dy = pgody[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)

          self.kp.append((sx, sy, l, (dx, dy), fhm[isy, isx]))

    return self.kp

  # Sub-pixel Accuracy
  @staticmethod
  def spa(p, neig):

    # Derivatives
    dx = (neig[1, 2]-neig[1, 0])/2.0
    dy = (neig[2, 1]-neig[0, 1])/2.0
    dxx = neig[1, 2]-2.0*neig[1, 1]+neig[1, 0]
    dyy = neig[2, 1]-2.0*neig[1, 1]+neig[0, 1]
    dxy = ((neig[2, 2]-neig[2, 0])-(neig[0, 2]-neig[0, 0]))/4.0

    # First-order and Second-order derivative
    fd = np.array([[dx], [dy]])
    sd = np.array([[dxx, dxy], [dxy, dyy]])

    pm = -np.dot(np.linalg.inv(sd), fd)
    return (p[0]+pm[0], p[1]+pm[1])
  
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

        if kp[i][4] < cr*kp[j][4] and dis(i, j)<minr:
          minr = dis(i, j)

      newkp.append((kp[i][0], kp[i][1], kp[i][2], kp[i][3], minr))

    newkp.sort(key=lambda x: x[4], reverse=True)

    return newkp[:self.nf]

def main():
  img = cv2.imread('data/parrington/prtn00.jpg')

  fd = MSOP()
  kp = fd.detect(img)
  nkp = fd.anms()

  # Show detected key points
  for p in nkp:
    cv2.circle(img, (p[0]*2**p[2], p[1]*2**p[2]), 2*2**p[2], (0, 0, 255))
    cv2.line(img, (p[0]*2**p[2], p[1]*2**p[2]), ((p[0]+p[3][0]*2)*2**p[2], (p[1]+p[3][1]*2)*2**p[2]), (0, 255, 0), 2)

  cv2.imwrite('kp_prtn00.jpg', img)

  return kp, nkp

if __name__=='__main__':
  kp, nkp = main()
  
