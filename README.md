# Blur Detection
Blur Detection works using the total variance of the laplacian of an
image, this provides a quick and accurate method for scoring how blurry
an image is.

## Setup
This package has a number of dependencies. These can be installed by running: 

```
python -m venv .
Scripts/activate
pip install -U -r requirements.txt
```
This creates a virtual environment for the script to run in so that it won't conflict with other scripts.

## Usage
The repository has a script, `process.py` which lets us run on single images or directories of images. The blur detection method is highly dependent on the size of the image being processed. To get consistent scores we fix the image size to HD, to disable this use  `--variable-size`. The script has options to, 

```bash
# run on a single image
python process.py -i input_image.png

# run on a directory of images
python process.py -i input_directory/ 

# or both! 
python process.py -i input_directory/ other_directory/ input_image.png
```

In addition to logging whether an image is blurry or not, we can also,

```bash
# save this information to json
python process.py -i input_directory/ -s results.json

# display blur-map image
python process.py -i input_directory/ -d

# move images into subfolders based on their evaluation
python process.py -i input_directory -m
```
The saved json file has information on how blurry an image is, the higher the value, the less blurry the image.

```json
{
    "images": ["/Users/demo_user/Pictures/Flat/"],
    "fix_size": true,
    "results": [
        {
            "blurry": false,
            "input_path": "/Users/demo_user/Pictures/Flat/IMG_1666.JPG",
            "score": 6984.8082115095549
        },
    ],
    "threshold": 100.0
}
```
# History
## Changes from parent branch
This was forked from [Will Brennan's BlurDetection2](https://github.com/WillBrennan/BlurDetection2). 
The major changes from that version are:
* Support for raw files via RawPy
  * Note: this has only been tested with NEF and CR2 files!
* A pre-filter of images with bilinear filtering, as suggested by OpenCV docs.
* An additional flag that causes images to be moved into subfolders based on their score (blurry/semi_blurry)
  * Note: This hasn't been tested extensively and may interact poorly with having multiple input paths. 
* A summary section after everything has been processed that shows statistics for the run.
* Bugfixes
  * A fix for the scaling system to increase the range of possible input image sizes
  * A fix to file detection that prevents files from being processed twice

## Citations
This is based upon the blogpost [Blur Detection With Opencv](https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/) by Adrian Rosebrock.

![Blur Mask Demo](https://raw.githubusercontent.com/WillBrennan/BlurDetection2/master/docs/demo.png)
