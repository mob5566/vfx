'''
# panorama.py -- Build panomara image with given image locations and focal length

* for digital visual effects homework2
* generate a panorama image by <image_list_file>
* with each row is image location and its focal length followed
* contents
  * cylindrical projection

'''

from __future__ import print_function

import os
import sys
from itertools import combinations
from itertools import product

import cv2
import numpy as np
from scipy.spatial import KDTree

from msop import MSOP

def dfs(vis, imgs, M, edge, u, con, bound, accM):
  con.append(u)
  vis.add(u)
  (h, w) = imgs[u].shape[:2]
  M[u] = accM

  for (x, y) in product((0, w), (0, h)):
    p = np.dot(accM[:2], [[x], [y], [1]])

    bound[0] = min(bound[0], p[0])
    bound[1] = min(bound[1], p[1])
    bound[2] = max(bound[2], p[0])
    bound[3] = max(bound[3], p[1])
  
  for (v, vM) in edge[u]:
    if v not in vis:
      dfs(vis, imgs, M, edge, v, con, bound, np.dot(accM, vM))

class Panorama(object):
  def process(self, imglistfile, outdir, dedrift=True):

    # Create directory
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    if not os.path.exists(imglistfile):
      print("{} doesn't exist!".format(imglistfile))
      sys.exit(-1)

    # Initialize all images with focal length
    print('Loading images...')
    self.fl = []
    self.imgs = []
    self.imgfn = []
    self.dedrift = dedrift

    with open(imglistfile, 'rb') as f:
      for line in f.readlines():
        imgfn, fl = line.split()
        fl = float(fl)
        
        self.fl.append(fl)
        self.imgs.append(Panorama.cylin_project(cv2.imread(imgfn), fl))
        self.imgfn.append('{}/cylin_{}.jpg'.format(outdir, len(self.imgs)))
        cv2.imwrite(self.imgfn[-1], self.imgs[-1])

    # Construct the descriptor of each image and k-d tree
    print('Constructing features...')
    feat_detect = MSOP(pyrLevel=5, numFeat=1000)
    self.info = []
    
    for fn in self.imgfn:
      kp, desp = feat_detect.detectAndDescribe(fn)
      self.info.append((kp, desp, KDTree(desp)))

    # Match images
    print('Matching images...')
    self.edge = [[] for i in xrange(len(self.imgs))]

    for ((i, infoA), (j, infoB)) in combinations(enumerate(self.info), 2):
      ismatch, (matchkp, M) = MSOP.imageMatch(infoA, infoB)

      if ismatch:
        M = np.append(M, [[0, 0, 1]], axis=0)
        self.edge[i].append((j, M))
      
    # Detect panoramas and construct connected components
    print('Detecting connected images...')
    self.connected = []
    self.M = {}
    vis = set()

    for i in xrange(len(self.imgs)):
      if i not in vis:
        con = []
        bound = np.array([0, 0, 0, 0])
        dfs(vis, self.imgs, self.M, self.edge, i, con, bound, np.eye(3))
        self.connected.append((con, bound))

    # Construct panorama for each connected components
    print('Constructing panoramas...')
    for i, (con, bound) in enumerate(self.connected):
      if len(con)<=1: continue

      baseM = np.array([[1, 0, -bound[0]], [0, 1, -bound[1]], [0, 0, 1]])
      nw = np.ceil(bound[2]-bound[0]).astype(int)
      nh = np.ceil(bound[3]-bound[1]).astype(int)
      baseImg = np.zeros((nh, nw, 3), dtype=np.uint8)

      ceny = []

      for u in con:
        self.M[u] = np.dot(baseM, self.M[u])
        h, w = self.imgs[u].shape[:2]
        ceny.append(np.dot(self.M[u], [[w/2], [h/2], [1]])[1])

      # Dedrift by average height
      if self.dedrift:
        meany = np.mean(ceny)
        for j, u in enumerate(con):
          self.M[u] = np.dot([[1, 0, 0], [0, 1, meany-ceny[j]], [0, 0, 1]], self.M[u])

      lx, rx = (None, None)
      
      for u in con:
        newImg = cv2.warpAffine(self.imgs[u], self.M[u][:2], (nw, nh))
        cv2.imwrite('{}/{}.jpg'.format(outdir, u), newImg)

        newMask = np.logical_not(np.all(newImg<=13, axis=2))
        mask = np.logical_not(np.all(baseImg<=13, axis=2))
        andMask = np.logical_and(mask, newMask)
        xorMask = np.logical_and(np.logical_not(mask), newMask)

        h, w = self.imgs[u].shape[:2]
        x1 = np.dot(self.M[u], [[0], [0], [1]])[0]
        x2 = np.dot(self.M[u], [[w], [0], [1]])[0]
        if x1 > x2:
          tmp = x2
          x2 = x1
          x1 = tmp

        if lx is not None:

          if lx < x2:
            a = lx
            b = x2
            A = newImg
            B = baseImg
          elif x1 < rx:
            a = x1
            b = rx
            A = baseImg
            B = newImg

          row, col = np.where(andMask)
          for (y, x) in zip(row, col):
            alpha = (x-a)/(b-a)
            baseImg[y, x] = A[y, x]*(1-alpha)+B[y, x]*alpha

          lx = min(lx, x1)
          rx = max(rx, x2)

        else:
          lx = x1
          rx = x2

        baseImg[xorMask] = newImg[xorMask]

      cv2.imwrite('{}/pano_{}.jpg'.format(outdir, i), baseImg)

    print('Done!')

  '''
  Static methods
  '''
  # Cylindrical projection
  @ staticmethod
  def cylin_project(img, f):
    assert(type(img)==np.ndarray and len(img.shape)>=2)

    h, w = img.shape[:2]

    x = np.arange(w).reshape(1, -1)
    for r in xrange(h-1):
      x = np.append(x, np.arange(w).reshape(1, -1), axis=0)

    y = np.arange(h).reshape(-1, 1)
    for c in xrange(w-1):
      y = np.append(y, np.arange(h).reshape(-1, 1), axis=1)

    xm = f*np.tan((x-w/2)/f)+w/2
    ym = (y-h/2)*np.sqrt((np.tan((x-w/2)/f)**2+1))+h/2

    return cv2.remap(img, xm.astype(np.float32), ym.astype(np.float32), cv2.INTER_LINEAR)[:, 10:-10]

def main(imgname):
  pass 

if __name__=='__main__':
  if len(sys.argv) == 3:
    pano = Panorama()
    pano.process(sys.argv[1], sys.argv[2], False)
