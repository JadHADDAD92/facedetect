""" extract frames containing faces from video file
"""
from decimal import Decimal as D
from os import listdir
from pathlib import Path
from shutil import move
from subprocess import Popen

import click
import cv2
from tqdm import tqdm

from common.facedetector import FaceDetector

dirname = Path(__file__).parent

imgExtensions = [".jpg", ".jpeg", ".png"]

def processDirectory(directoryPath: Path):
    """ process files in directory
    """
    keyframesPath = directoryPath/'keyframes'
    filesPaths = [f for f in listdir(str(keyframesPath)) if (keyframesPath/f).is_file()
                  and Path(f).suffix in imgExtensions]
    faceDetector = FaceDetector(prototype='models/deploy.prototxt.txt',
                                model='models/res10_300x300_ssd_iter_140000.caffemodel')
    
    dirNames = [ directoryPath/'faces', directoryPath/'non_faces' ]
    for dirName in dirNames:
        if dirName.exists() is False:
            dirName.mkdir()
    nonFacesCounter = 0
    facesCounter = 0
    filesTQDM = tqdm(filesPaths)
    for filePath in filesTQDM:
        filePath = str(keyframesPath/filePath)
        im = cv2.imread(filePath)
        
        if faceDetector.detect(im):
            move(filePath, str(dirNames[0]))
            facesCounter = facesCounter + 1
        else:
            move(filePath, str(dirNames[1]))
            nonFacesCounter = nonFacesCounter + 1
        filesTQDM.set_description(f'face_frames:{facesCounter}, '
                                  f'non_face_frames:{nonFacesCounter}')
        filesTQDM.update(1)
    filesTQDM.close()
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
@click.option('--percentage', default=1.5, type=D,
              help="percentage threshold of difference between two consecutive frames")
# pylint: disable=invalid-name
def processFile(file: str, output_directory: str, percentage: D):
    """ process video file
    """
    percentage = percentage/100
    outputPath = Path(output_directory)
    
    keyframesDir = outputPath/'keyframes'
    if not keyframesDir.exists():
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
    
    processDirectory(outputPath)

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
    processFile()
