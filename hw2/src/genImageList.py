"""
# genList.py -- List files in a given directory

generate a file containing path of files in <image-location>

"""

from __future__ import print_function

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

    f.write('{}\n'.format(afn))

