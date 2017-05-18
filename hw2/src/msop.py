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
import cPickle as pickle

import cv2
import numpy as np
from scipy.spatial import KDTree

from itertools import product

# Epsilon
eps = 1e-8

# Transformation
def getTranslationTransform(pa, pb):
  assert(type(pa)==np.ndarray and pa.shape==(1,2))
  assert(type(pb)==np.ndarray and pb.shape==(1,2))

  x, y = pb[0]-pa[0]

  return np.array([[1, 0, x], [0, 1, y]])

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
          # sx, sy = MSOP.spa((x, y), nei)
          # isx = int(round(sx))
          # isy = int(round(sy))
          isx = x
          isy = y
          dx = pgodx[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)
          dy = pgody[isy, isx]/np.sqrt(pgodx[isy, isx]**2+pgody[isy, isx]**2)

          self.kp.append((x, y, l, (dx, dy), fhm[isy, isx]))

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
    coef = []
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
      patch = cv2.resize(patch, featSize).astype(np.float)
      patch = (patch-patch.mean())/(patch.std()+eps)
      
      desp.append(patch.reshape(-1))

    self.desp = np.array(desp)

    return self.desp

  # Find keypoints and compute descriptor
  def detectAndDescribe(self, img):
    feat_fn = img[:-4]+'_feat.pkl'
    
    if os.path.exists(feat_fn):
      with open(feat_fn, 'rb') as f:
        (kp, desp) = pickle.load(f)
    else:
      self.detect(cv2.imread(img))
      kp = self.anms()
      desp = self.descriptor()
      with open(feat_fn, 'wb') as f:
        pickle.dump((kp, desp), f)

    return kp, desp

  # Draw keypoints
  @staticmethod
  def drawKeypoints(img, kp, top=None, cc=(0, 0, 255), cl=(255, 0, 0), showScale=False):
    assert(len(cc)==3)
    cc = tuple(cc)
    assert(len(cl)==3)
    cl = tuple(cl)

    if not top:
      top = len(kp)

    for p in kp[:top]:
      x, y, l, (dx, dy) = p

      toi = lambda v: int(round(v))
      cx1 = toi(x*2**l)
      cy1 = toi(y*2**l)
      cx2 = toi((x+dx*20)*2**l)
      cy2 = toi((y+dy*20)*2**l)

      if showScale:
        cv2.circle(img, (cx1, cy1), 20*2**l, cc)
        cv2.line(img, (cx1, cy1), (cx2, cy2), cl, 1)
      else:
        cv2.circle(img, (cx1, cy1), 2, cc)
    
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

  # RANSAC
  @staticmethod
  def ransac(kpa, kpb, sample_n=3, p_inlier=0.8, P=0.99, inlier_thresh=10):
    pa = np.array([(x, y) for x, y, _, _ in kpa])
    pb = np.array([(x, y) for x, y, _, _ in kpb])
    n = len(pa)

    if n < 10:
      return None, None

    k = np.ceil(np.log(1-P)/np.log(1-p_inlier**sample_n)).astype(int)

    maxin = 0
    bestM = None
    inliers = None
    sample_mask = np.array([False]*n)
    sample_mask[:sample_n] = True

    for i in xrange(k):
      # Sample n points
      sample_mask = np.random.permutation(sample_mask)
      spa = pa[sample_mask]
      spb = pb[sample_mask]

      # Compute transformation
      # M = getTranslationTransform(spb.astype(np.float32), spa.astype(np.float32))
      M = cv2.getAffineTransform(spb.astype(np.float32), spa.astype(np.float32))

      tpa = np.dot(M, np.append(pb, np.ones(n).reshape(-1, 1), axis=1).T).T
      inl = np.linalg.norm(pa-tpa, axis=1) < inlier_thresh

      if maxin < inl.sum():
        maxin = inl.sum()
        bestM = M.copy()
        inliers = inl.copy()

    return bestM, inliers

  # Image MSOP matching
  @staticmethod
  def imageMatch(infoA, infoB):
    kpa, despa, kdta = infoA
    kpb, despb, kdtb = infoB
    matchkp = []
    f = 0.65

    for ki in xrange(len(despa)):
      (nn1, nn2), (kb, _) = kdtb.query(despa[ki], 2)

      if nn1 < f*nn2:
        matchkp.append((ki, kb))

    if len(matchkp) < 10:
      return False, (None, None)
    
    mkpa = []
    mkpb = []

    for (ka, kb) in matchkp:
      mkpa.append(kpa[ka])
      mkpb.append(kpb[kb])

    M, inliers = MSOP.ransac(mkpa, mkpb)

    if M is None:
      return False, (None, None)

    newmatchkp = []

    for v, kp in zip(inliers, matchkp):
      if v: newmatchkp.append(kp)
    
    return (inliers.sum() > 5.9+0.22*len(matchkp)), (newmatchkp, M)

def testMSOP(imgname):
  # Function for MSOP testing
  img = cv2.imread(imgname)

  fd = MSOP(pyrLevel=3)
  kp, desp = fd.detectAndDescribe(imgname)

  MSOP.drawKeypoints(img, kp, top=100, showScale=True)

  cv2.imwrite(os.path.basename(imgname), img)

  return kp, desp

def testMatch(imn1, imn2):
  # Function for image matching testing
  img1 = cv2.imread(imn1)
  img2 = cv2.imread(imn2)

  fd = MSOP(pyrLevel=3, numFeat=1000)
  kp1, desp1 = fd.detectAndDescribe(imn1)
  kp2, desp2 = fd.detectAndDescribe(imn2)

  kdt1 = KDTree(desp1)
  kdt2 = KDTree(desp2)
  ismatch, (matchkp, M) = MSOP.imageMatch((kp1, desp1, kdt1), (kp2, desp2, kdt2))

  matA = set([u for u, v in matchkp])
  matB = set([v for u, v in matchkp])

  kp1m = []
  kp1n = []
  kp2m = []
  kp2n = []

  for i in xrange(len(kp1)):
    if i in matA:
      kp1m.append(kp1[i])
    else:
      kp1n.append(kp1[i])
  
  for i in xrange(len(kp2)):
    if i in matB:
      kp2m.append(kp2[i])
    else:
      kp2n.append(kp2[i])

  MSOP.drawKeypoints(img1, kp1m, cc=(0, 255, 0))
  MSOP.drawKeypoints(img1, kp1n, cc=(0, 0, 255))
  MSOP.drawKeypoints(img2, kp2m, cc=(0, 255, 0))
  MSOP.drawKeypoints(img2, kp2n, cc=(0, 0, 255))

  cv2.imwrite(os.path.basename(imn1), img1)
  cv2.imwrite(os.path.basename(imn2), img2)

if __name__=='__main__':
  if len(sys.argv)==2:
    kp, desp = testMSOP(sys.argv[1])
  
  if len(sys.argv)==3:
    testMatch(sys.argv[1], sys.argv[2])
