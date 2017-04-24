'''
# msop.py -- Multi-Scale Oriented Patches feature

* for digital visual effects homework2
* Multi-Scale Oriented Patches feature of feature matching
* Reference from M. Brown, R. Szeliski, and S. Winder
* "Multi-Image Matching using Multi-Scale Oriented Patches"

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
  def __init__(self, numFeat=500, pyrLevel=3, fhmt=10.0, samplePatchSize=8, sampleSpace=5):
    self.nf = numFeat           # Number of features
    self.pyrl = pyrLevel        # Number of levels of pyramid
    self.ft = fhmt              # Harmonic mean threshold
    self.kp = None              # Keypoints
    self.sps = samplePatchSize  # Sample Patch Size
    self.ss = sampleSpace       # Sample space
  
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
      pdx = cv2.Sobel(p, cv2.CV_64F, 1, 0, ksize=1)
      pdy = cv2.Sobel(p, cv2.CV_64F, 0, 1, ksize=1)

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
      for x, y in product(range(30, cw-30), range(30, ch-30)):
        nei = fhm[y-1:y+2, x-1:x+2]
        if fhm[y, x]>=self.ft and np.all(fhm[y, x]>=nei.reshape(-1)):
          sx, sy = MSOP.spa((x, y), nei)
          isx = int(round(sx))
          isy = int(round(sy))
          dx = pgodx[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)
          dy = pgody[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)

          self.kp.append((sx, sy, l, (dx, dy), fhm[isy, isx]))

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

        if kp[i][4] < cr*kp[j][4] and dis(i, j)<minr:
          minr = dis(i, j)

      newkp.append((kp[i][0], kp[i][1], kp[i][2], kp[i][3], minr))

    newkp.sort(key=lambda x: x[4], reverse=True)

    self.kp = [(x, y, l, d) for x, y, l, d, r in newkp[:self.nf]]
    return self.kp

  # Compute the descriptor of features
  def descriptor(self):
    desp = []
    gk = cv2.GaussianBlur
    ksize = (5, 5)
    ppyramid = [gk(gk(pyr.copy(), ksize, 1.0), ksize, 1.0) for pyr in self.pyramid]

    patchSize = (self.sps*self.ss, self.sps*self.ss)
    featSize = (self.sps, self.sps)

    # Get the patch of each keypoint and rotate it with main gradient
    for i, p in enumerate(self.kp):
      x, y, l, (dx, dy) = p
      pimg = ppyramid[l]
      ch, cw = pimg.shape
      theta = np.arctan2(-dy, dx)*180/np.pi
      
      M = cv2.getRotationMatrix2D((x, y), -theta, 1.)
      trans = cv2.warpAffine(pimg, M, (cw, ch))
      patch = cv2.getRectSubPix(np.round(trans).astype(np.uint8), patchSize, (x, y))
      patch = cv2.resize(patch, featSize)

      des = patch.reshape(-1).astype(np.float)
      des = (des-des.mean())/(des.std()+eps)

      desp.append(des)

    self.desp = np.array(desp)
    return self.desp

  # Find keypoints and compute descriptor
  def detectAndDescribe(self, img):
    self.detect(img)
    kp = self.anms()
    desp = self.descriptor()

    return kp, desp

  # Draw keypoints
  @staticmethod
  def drawKeypoints(img, kp, top=30, cc=(0, 0, 255), cl=(255, 0, 0)):
    assert(len(cc)==3)
    cc = tuple(cc)
    assert(len(cl)==3)
    cl = tuple(cl)
    for p in kp[:top]:
      x, y, l, (dx, dy) = p

      toi = lambda v: int(round(v))
      cx1 = toi(x*2**l)
      cy1 = toi(y*2**l)
      cx2 = toi((x+dx*20)*2**l)
      cy2 = toi((y+dy*20)*2**l)

      cv2.circle(img, (cx1, cy1), 20*2**l, cc)
      cv2.line(img, (cx1, cy1), (cx2, cy2), cl, 1)
    
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
    return (p[0]+pm[0, 0], p[1]+pm[1, 0])

def main(imgname):
  # Main function for MSOP testing
  img = cv2.imread(imgname)

  fd = MSOP(pyrLevel=3)
  kp, desp = fd.detectAndDescribe(img)

  MSOP.drawKeypoints(img, kp, top=100)

  cv2.imwrite(os.path.basename(imgname), img)

  return kp, desp

if __name__=='__main__':
  if len(sys.argv)==2:
    kp, desp = main(sys.argv[1])
  else:
    kp, desp = main('data/parrington/prtn00.jpg')
  
