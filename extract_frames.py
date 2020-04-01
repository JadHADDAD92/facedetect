""" extract frames containing faces from video file
"""
from decimal import Decimal as D
from os import listdir
from pathlib import Path
from shutil import Error, move
from subprocess import Popen

import click
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
    # try to detect faces using different haar cascades
    for cascade in CASCADES:
        fc = cv2.CascadeClassifier(str(dirname/'haarcascades'/cascade))
        faces = fc.detectMultiScale(image)
        if len(faces) > 0:
            return True
    return False


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

@click.command(context_settings=dict(show_default=True),
               help="extract frames from video file and classify them in two separate "
                    "directories: faces and non_faces")
@click.argument('file')
@click.argument('output_directory')
@click.option('--percentage', default=1.5,
              help="percentage threshold of difference between two consecutive frames")
# pylint: disable=invalid-name
def processFile(file, output_directory, percentage):
    """ process video file
    """
    percentage = D(percentage)/D(100)
    outputPath = Path(output_directory)
    
    keyframesDir = outputPath/'keyframes'
    if keyframesDir.exists() is False:
        keyframesDir.mkdir()
    
    extractFrames = ['ffmpeg',
                     '-i', file,
                     '-qscale:v', '2',
                     '-vf', 'select=gt(scene\\,%s)'%percentage,
                     '-vsync', 'vfr',
                     str(keyframesDir)+'/thumb%04d.jpg',
                     '-loglevel', 'error'
                    ]
    
    extractFramesP = Popen(extractFrames)
    print('extracting frames from %s'%file)
    extractFramesP.wait()
    
    print('classifying images in faces/non_faces...')
    processDirectory(outputPath)

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    processFile()
