"""
Module to find and highlight foxes in pictures
"""

from typing import List
import cv2


def find(image_path: str,
         output_path: str = None,
         scale_factor: float = 1.31,
         min_neighbors: int = 10,
         min_size: tuple = (50, 50)) -> List:
    """
    Search for a fox in the provided image.
    If the output path is not null, highlight the found
    foxes in the picture, and save to the path. Return
    the location of the found foxes.

    Parameter
    ---------
    image_path: str
        The path of the image to process.
    output_path: str, optional
        The path to save the output image.

    Return
    ------
    list
        List of the found foxes location.
    """
    img = cv2.imread(image_path)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    stop_data = cv2.CascadeClassifier('cascade.xml')

    found = stop_data.detectMultiScale(img_gray,
                                       scaleFactor=scale_factor,
                                       minNeighbors=min_neighbors,
                                       minSize=min_size)

    if output_path is not None:
        for (x, y, width, height) in found:

            # We draw a green rectangle around
            # every recognized sign
            cv2.rectangle(img, (x, y),
                          (x + height, y + width),
                          (0, 255, 0), 2)
        cv2.imwrite(output_path, img)

    return found
