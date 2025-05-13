import os
import re

import cv2
import numpy as np

current_path = os.path.dirname(__file__)


def gen_mask_image(resource, **kwargs):
    points = kwargs.get('mask_points', [])
    reverse = kwargs.get('reverse', False)
    if len(points) <= 0:
        return None
    origin_image = cv2.imread(resource)
    origin_shape = origin_image.shape
    output_path = re.sub(r"\.((jp?g)|(png)|(bmp)|(webp))", r"_mask.\1", resource)
    print(output_path)
    for point in points:
        if not 0 <= point[0] <= origin_shape[1]:
            raise Exception("x坐标超过图片范围")
        if not 0 <= point[1] <= origin_shape[0]:
            raise Exception("y坐标超过图片范围")
    # exit(0)
    if not reverse:
        im = np.zeros([origin_shape[0], origin_shape[1]], dtype=np.uint8)
        sss = np.array([points], dtype=np.int32)
        cv2.fillPoly(im, sss, [255, 255])
        cv2.imwrite(output_path, im)

    else:
        im = 255 - np.zeros([origin_shape[0], origin_shape[1]], dtype=np.uint8)
        sss = np.array([points], dtype=np.int32)
        cv2.fillPoly(im, sss, 0)
        cv2.imwrite(output_path, im)
    return output_path


def TEST():
    points = [
        [8060.3, 100],
        [1000, 100],
        [8060.3, 1000],
        [1000, 1000]
    ]
    gen_mask_image("../test_data/cat.jpg", mask_points=points,reverse=False)


if __name__ == "__main__":
    TEST()
