# frameExtractor
this is a small sub-project aimed to extract valuable and meaningful frames (by discarding very similar consecutive frames) from video files.
can be used for creating a dataset for machine learning. </br>
extract only frames that contain changes (specified in --percentage) from videos, and classify them in two directories: faces / non_faces

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
