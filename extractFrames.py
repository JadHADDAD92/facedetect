""" extract frames containing faces from video file
"""
import argparse
from decimal import Decimal as D
from os import listdir
from pathlib import Path
from shutil import move, Error
from subprocess import Popen
import cv2

dirname = Path(__file__).parent

CASCADES = [
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml'
]

imgExtensions = [".jpg", ".jpeg", ".png"]

 # pylint: disable=len-as-condition
def faceDetected(image):
    """ detect faces in image
    """
    image = cv2.equalizeHist(image)
    for cascade in CASCADES:
        fc = cv2.CascadeClassifier(str(dirname/'haarcascades'/cascade))
        faces = fc.detectMultiScale(image)
        if len(faces) > 0:
            return True
    return False

def processFile(filePath, outputPath, percentage):
    """ process video file
    """
    percentage = D(percentage)/D(100)
    outputPath = Path(outputPath)
    
    keyframesDir = outputPath/'keyframes'
    if keyframesDir.exists() is False:
        keyframesDir.mkdir()
    
    extractFrames = ['ffmpeg',
                     '-i', filePath,
                     '-vf', 'select=gt(scene\\,%s)'%percentage,
                     '-vsync', 'vfr',
                     str(keyframesDir)+'/thumb%04d.jpg',
                     '-loglevel', 'error'
                    ]
    
    extractFramesP = Popen(extractFrames)
    print('extracting frames from %s'%filePath)
    extractFramesP.wait()
    
    print('classifying images in faces/non_faces...')
    processDirectory(outputPath)

def processDirectory(directoryPath):
    """ process files in directory
    """
    keyframesPath = directoryPath/'keyframes'
    filesPaths = [f for f in listdir(str(keyframesPath)) if (keyframesPath/f).is_file()
                  and Path(f).suffix in imgExtensions]
    
    numOfFiles = len(filesPaths)
    
    dirNames = [ directoryPath/'faces', directoryPath/'non_faces' ]
    for dirName in dirNames:
        if dirName.exists() is False:
            dirName.mkdir()
    counter = 1
    nonFacesCounter = 0
    facesCounter = 0
    for filePath in filesPaths:
        filePath = str(keyframesPath/filePath)
        im = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        try:
            if faceDetected(im):
                move(filePath, str(dirNames[0]))
                facesCounter = facesCounter + 1
            else:
                move(filePath, str(dirNames[1]))
                nonFacesCounter = nonFacesCounter + 1
        except Error as error:
            print(error)
            print('skipping file')
        
        print("\r%d/%d"%(counter, numOfFiles), end='\r')
        counter = counter +1
    print('%d images containing faces detected \n%d images not containing faces detected'%
          (facesCounter, nonFacesCounter))
    try:
        keyframesPath.rmdir()
    except OSError as error:
        print(error)
        print('couldn\'t delete temporary directory %s'%keyframesPath)

parser = argparse.ArgumentParser()
parser.add_argument('file')
parser.add_argument('outputDirectory')
parser.add_argument('--percentage', default=1.5,
                    help="percentage threshold of difference between"
                         " two consecutive frames")
args = parser.parse_args()
processFile(args.file, args.outputDirectory, args.percentage)
