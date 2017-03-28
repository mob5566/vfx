"""
# genImageList.py -- Generate file of image list for hdr.py

* for digital visual effects homework1 -- `hdr.py`
* generate a file with <image-location> and <exposure-time>
  splits by space per line

"""

from __future__ import print_function

import PIL.Image
import PIL.ExifTags
import os
import sys

# The default output file name
outf = os.path.join('.', 'imglist.txt')

if len(sys.argv)!=2 and len(sys.argv)!=3 or not os.path.isdir(sys.argv[1]):
  print('    Usage: python genImageList.py <directory of images> [output file]')
  sys.exit(-1)
else:
  imgsdir = sys.argv[1]

  if len(sys.argv)==3:
    outf = sys.argv[2]

# Get the name of images and output file name
fns = os.listdir(imgsdir)

# Write the information to the output file
with open(outf, 'wb') as f:
  for fn in fns:
    afn = os.path.abspath(os.path.join(imgsdir, fn))
    img = PIL.Image.open(afn)

    exif = {
      PIL.ExifTags.TAGS[k]: v
      for k, v in img._getexif().items()
      if k in PIL.ExifTags.TAGS
    }

    exptime = float(exif['ExposureTime'][0])/exif['ExposureTime'][1]

    f.write('{} {}\n'.format(afn, exptime))

