import cv2
import sys
import argparse
import os

dirname = os.path.dirname(__file__)

CASCADES = [
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml'
]

def detectFace(filepath):
    """ detect faces in image
    """
    im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if im is None:
        print("cannot load input image {}".format(filepath))
    im = cv2.equalizeHist(im)
    for cascade in CASCADES:
        fc = cv2.CascadeClassifier(os.path.join(dirname,'haarcascades',cascade))
        faces = fc.detectMultiScale(im)
        if len(faces) > 0:
            print(faces)
            sys.exit(0)
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument('file')
args = parser.parse_args()
detectFace(args.file)
