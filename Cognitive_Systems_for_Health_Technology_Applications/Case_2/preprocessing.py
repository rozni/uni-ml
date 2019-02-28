"""Some preprocessing functions I've been using again and again..."""

__all__ = []

from multiprocessing.dummy import Pool as ThreadPool

import os
import time
import numpy as np
import cv2 as cv

def _export(func):
    """
    A function decorator to export its name.

    If __all__ is defined only the names inside it are visible
    when the module is imported.
    Otherwise, all names are visible - e.g. ThreadPool, np,
    would be visible for the importer.
    """
    global __all__
    __all__.append(func.__name__)
    return func


@_export
def crop_eye(
        img,
        brightness_threshold=20,
        min_hot_frac=0.03,
):
    """
    Expects OpenCV image array (BGR), returns cropped eye

    brightness_threshold -- do not count a pixel if its value is below
    the threshold, consider those above "hot" (default 20)

    min_hot_frac -- crop where at least this fraction of the
    maximum hot pixel count is hot (default 0.03)
    """

    # background is black, so we can find the eye
    # based on brightness, say it's an eye pixel
    # if it's brighter than `brightness_threshold`,
    # otherwise, consider it background
    _, img_bin = cv.threshold(
        cv.cvtColor(img, cv.COLOR_BGR2GRAY),
        thresh=brightness_threshold,
        maxval=1,
        type=cv.THRESH_BINARY
    )

    # horizontal and vertical "histogram"
    hor_sum = img_bin.sum(0)
    ver_sum = img_bin.sum(1)

    # keep if at least the given fraction of pixels is nonzero
    hor_indices = hor_sum > (min_hot_frac*hor_sum.max())
    ver_indices = ver_sum > (min_hot_frac*ver_sum.max())

    # slices are muuuch faster than indexing
    h_min, h_max = np.nonzero(hor_indices)[0][[0, -1]]
    v_min, v_max = np.nonzero(ver_indices)[0][[0, -1]]
    cropped = img[v_min:v_max+1, h_min:h_max+1, :]

    return cropped


@_export
def pad_to_square(img, color=np.zeros(3)):
    """Adds padding of `color` around the image
    to make it square, keeping it centered.
    (height and widht might differ by 1 pixel)"""

    height, width, _ = img.shape
    padding = (width - height) // 2

    height_padding = max(padding, 0)
    width_padding = max(-padding, 0)

    square = cv.copyMakeBorder(
        img,
        top=height_padding,
        bottom=height_padding,
        left=width_padding,
        right=width_padding,
        borderType=cv.BORDER_CONSTANT,
        value=color
    )

    return square


@_export
def image_resize(img, width, height, keep_aspect_ratio=False, interpolation=cv.INTER_LANCZOS4):
    """
    Resize to given `width` and `height`, if `keep_aspect_ratio`
    then the final image will be contained withing the box.

    For `interpolation`, choose from OpenCV interpolations.
    """

    if keep_aspect_ratio:
        orig_h, orig_w = img.shape[:2]

        suggested_height = orig_h * width // orig_w
        if suggested_height <= height:
            height = suggested_height
        else:
            width = orig_w * height // orig_h

    resized = cv.resize(img, dsize=(width, height), interpolation=interpolation)

    return resized


@_export
def process_img_files(
        src_dir,
        dst_dir,
        img_names,
        dst_names,
        process,
        verbose=True
):
    """
    Opens every image from `img_names` in `src_dir` applies `process`
    and saves the result to the corresponding name (`dst_names`) in `dst_dir`.

    Runs in parallel and creates the directory structure if needed.
    Indicates the progress if `verbose`.

    Examples:

    # copies all images JPEGs from /A to /B/C, preserving structure
    >>> images = glob('/A/**/*.jpe?g')
    >>> process_img_files('/A', '/B/C', images, images, lambda img: img)

    # converts eye.jpg to grayscale and saves it as a PNG file
    >>> to_grayscale = lambda img: cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    >>> process_img_files('','', 'eye.jpg', 'eye.png', to_grayscale)
    """

    def process_img(img_in, img_out):
        path_in = os.path.join(src_dir, img_in)
        path_out = os.path.join(dst_dir, img_out)

        img = cv.imread(path_in)
        img = process(img)

        success = cv.imwrite(path_out, img)

        if not success:
            new_dir = os.path.dirname(path_out)
            os.makedirs(new_dir, exist_ok=True)

            success = cv.imwrite(path_out, img)
            if not success:
                print(path_in, 'to', path_out, 'failed')

    with ThreadPool() as pool:
        start_time = time.time()
        print('Started:', time.ctime(start_time))

        # interruption in Jupyter misbehaves for starmap
        # but this lazy approach works + can show the progress
        it = pool.imap(lambda ii: process_img(*ii), zip(img_names, dst_names))

        try:
            num_images = len(img_names)
            progress_delta = num_images // 1000 or num_images // 100
            last_time = time.time()
            last_count = 0

            for count, _ in enumerate(it):

                if verbose and last_count + progress_delta < count:
                    cur_time = time.time()
                    img_rate = (count - last_count) / (cur_time - last_time)
                    progress = 100 * count / num_images
                    eta = (num_images - count) * (cur_time - start_time) / count

                    print(
                        f'{img_rate:6.2f} images/second    '
                        f'{progress:6.2f}% done    '
                        f'({int(eta)} seconds remaining)',
                        end='\r'
                    )

                    last_count = count
                    last_time = cur_time

        except KeyboardInterrupt:
            print(f'terminating ({time.ctime()})')
            pool.terminate()

        pool.close()
        pool.join()

        end_time = time.time()
        print('Total elapsed time (s):', end_time - start_time)
        print(f'DONE! :) ({time.ctime(end_time)})')
