'''
# tonemap.py -- HDR image tone mapping

* for digital visual effects homework1
* generate a tone mapping image by an exr HDR image

'''

import sys

import cv2
import numpy as np

# Normalize image
def imgscaling(inimg):
  return (inimg-inimg.min())/(inimg.max()-inimg.min())

# Tone mapping 
def tonemap(img, basecon, spasig, ransig, gammacor):
  width = img.shape[1]
  height = img.shape[0]

  gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  lgimg = np.log(gimg)

  # Multiscale decomposition
  base = cv2.bilateralFilter(lgimg, -1, ransig, spasig)
  detail = lgimg-base

  # Contrast reduction
  base = base*basecon/(base.max()-base.min())

  # Reconstruct intensity
  nimg = np.exp(base+detail)

  # Recompose color image
  cimg = (nimg/gimg).reshape(height, width, 1)*img

  # Normalize 
  for i in xrange(cimg.shape[2]):
    cimg[..., i] = imgscaling(cimg[..., i])
  
  nimg = imgscaling(nimg)

  # Gamma correction
  cimg = np.power(cimg, gammacor)
  nimg = np.power(nimg, gammacor)

  return cimg, base, detail, nimg

# For GUI functions
def nothing(event):
  pass

# Default output file name
outImage = 'out'

# Loading the input image
if len(sys.argv)!=2 and len(sys.argv)!=3:
  print('    Usage: python tonemap.py <image.exr> [output image]')
  sys.exit(-1)
else:
  inFile = sys.argv[1]

  if len(sys.argv)==3:
    outImage = sys.argv[2]

img = cv2.imread(inFile, -1)
width = img.shape[1]
height = img.shape[0]

# OpenCV GUI settings
wn = 'Tone Mapping'
cv2.namedWindow(wn)
bc = 'Base contrast'
sks = 'Spatial kernel sigma' 
rks = 'Range kernel sigma'
gm = 'Gamma correction (*0.1)'

cv2.createTrackbar(bc, wn, 5, 50, nothing)
cv2.createTrackbar(sks, wn, 2, 100, nothing)
cv2.createTrackbar(rks, wn, 2, 20, nothing)
cv2.createTrackbar(gm, wn, 10, 100, nothing)

# Scaling down input image for preview
while width > 1000:
  width /= 2
  height /= 2

# Show message
print("Press 't' to do tone mapping")
print("Press 'r' to restore input image")
print("Press 's' to save HDR image")
print("Press 'q' to quit program")

simg = cv2.resize(img, (width, height))
oimg = img
base = np.zeros(img.shape[:2])
detail = np.zeros(img.shape[:2])
nimg = np.zeros(img.shape[:2])

img = np.power(imgscaling(img), 0.45)

while True:
  cv2.imshow(wn, simg)
  key = cv2.waitKey(30)&255

  # Tone mapping the input image
  if key==ord('t'):
    print('Tone mapping...')
    # Get parameters
    baseContrast = float(cv2.getTrackbarPos(bc, wn))
    spatialsig = float(cv2.getTrackbarPos(sks, wn))
    rangesig = float(cv2.getTrackbarPos(rks, wn))
    gamma = float(cv2.getTrackbarPos(gm, wn))/10.

    # Tone mapping
    oimg, base, detail, nimg = tonemap(img, baseContrast, spatialsig, rangesig, gamma)
    oimg *= 255

    simg = cv2.resize(oimg, (width, height))
    print('Tone mapping done')

  # Save image
  if key==ord('s'):
    baseContrast = float(cv2.getTrackbarPos(bc, wn))
    spatialsig = float(cv2.getTrackbarPos(sks, wn))
    rangesig = float(cv2.getTrackbarPos(rks, wn))
    gamma = float(cv2.getTrackbarPos(gm, wn))/10.
    para = '_bc'+str(baseContrast)+'_ss'+str(spatialsig)+'_rs'+str(rangesig)+'_gm'+str(gamma)

    print('Saving image...')
    cv2.imwrite(outImage+para+'.tiff', oimg)
    cv2.imwrite(outImage+para+'_base.tiff', np.repeat(base.reshape(oimg.shape[0], -1, 1), 3, axis=2))
    cv2.imwrite(outImage+para+'_detail.tiff', np.repeat(detail.reshape(oimg.shape[0], -1, 1), 3, axis=2))
    cv2.imwrite(outImage+para+'_nimg.tiff', np.repeat(nimg.reshape(oimg.shape[0], -1, 1), 3, axis=2))
    print('Save done')

  # Restore simg
  if key==ord('r'):
    simg = cv2.resize(img, (width, height))

  # Quit
  if key==ord('q'):
    break

cv2.destroyAllWindows()
