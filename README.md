# frameExtractor
this is a small sub-project aimed to extract valuable and meaningful frames from video files.
can be used for creating a dataset for machine learning
extract only frames that contain changes (precised in --percentage) from videos, and classify them in two directories: faces / non-faces

# requirements
* opencv
* ffmpeg

# usage
`python3 extractFrames.py /path/to/file.mp4 path/to/output/ --percentage 5`

## positional arguments:
 * file
 * outputDirectory

## optional arguments:
 * --percentage PERCENTAGE (percentage threshold of difference between two
                           consecutive frames)
