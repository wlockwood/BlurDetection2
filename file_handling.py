import pathlib
import cv2
import rawpy
import numpy
import logging
from PIL import Image
from io import BytesIO

raw_types = ['.nef', '.cr2', '.cr3']
normal_types = ['.jpg', '.jpeg', '.png']

def find_images(image_paths, img_extensions=raw_types + normal_types):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file():
            logging.info(f"Found image file '{path}'")
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            logging.info(f"Found directory '{path}'")
            for img_ext in img_extensions:
                yield from path.glob(f'*{img_ext}')


def load_image(path):
    is_raw = any(e in path.lower() for e in raw_types)
    if is_raw:
        with rawpy.imread(path) as raw:
            # raises rawpy.LibRawNoThumbnailError if thumbnail missing
            # raises rawpy.LibRawUnsupportedThumbnailError if unsupported format
            thumb = raw.extract_thumb()

        if thumb.format == rawpy.ThumbFormat.JPEG:
            # thumb.data is already in JPEG format, save as-is
            bio = BytesIO(thumb.data)
            pilmage = Image.open(bio)
            out = cv2.cvtColor(numpy.array(pilmage), cv2.COLOR_RGB2BGR)

        elif thumb.format == rawpy.ThumbFormat.BITMAP:
            #cv2.cvtColor(numpy.array(pilmage), cv2.COLOR_RGB2BGR)
            out = thumb.data
    else:
        out = cv2.imread(str(path))
    if out is None:
        raise Exception(f"Failed to load image data from {path}")
    return out