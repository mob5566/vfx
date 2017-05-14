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

import cv2
import numpy as np
from scipy.spatial import KDTree

from msop import MSOP

class Panorama(object):
  def __init__(self, imglistfile):

    if not os.path.exists(imglistfile):
      print("{} doesn't exist!".format(imglistfile))
      sys.exit(-1)

    # Initialize all images with focal length
    self.fl = []
    self.imgs = []

    with open(imglistfile, 'rb') as f:
      for line in f.readlines():
        imgfn, fl = line.split()
        fl = float(fl)
        
        self.fl.append(fl)
        self.imgs.append(Panorama.cylin_project(cv2.imread(imgfn), fl))

    # Construct the descriptor of each image and k-d tree
    feat_detect = MSOP(pyrLevel=5, numFeat=1000)
    self.kp = []
    self.fd = []
    self.tree = []
    
    for img in self.imgs:
      kp, desp = feat_detect.detectAndDescribe(img)
      self.kp.append(kp)
      self.fd.append(desp)
      self.tree.append(KDTree(desp))

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

    return cv2.remap(img, xm.astype(np.float32), ym.astype(np.float32), cv2.INTER_LINEAR)

def main(imgname):
  pass 

if __name__=='__main__':
  if len(sys.argv) == 2:
    pano = Panorama(sys.argv[1])
