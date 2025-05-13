# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2


def compute_bbox_info(img, boxes):
    """ save text detection result one by one
    Args:
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        box area and percentage
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    box_info = []
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        # compute area
        box_area = cv2.contourArea(poly)

        # compute area precente
        img_area = img.shape[0] * img.shape[1]
        box_percentage = round((box_area / img_area) * 100, 5) 
        box_info.append((poly, box_area, box_percentage))

    return box_info


def compute_sum_bbox_pct(img, boxes):
    """ get total percentage box
    Args:
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        total box percentage
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    sum_box_pct = 0
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        # compute area
        box_area = cv2.contourArea(poly)

        # compute area precente
        img_area = img.shape[0] * img.shape[1]
        box_percentage = (box_area / img_area) * 100

        sum_box_pct += box_percentage

    return round(sum_box_pct, 5)


def compute_max_bbox_pct(img, boxes):
    """ get max percentage box
    Args:
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        max box percentage
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    max_box_pct = -1
    for i, box in enumerate(boxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        # compute area
        box_area = cv2.contourArea(poly)

        # compute area precente
        img_area = img.shape[0] * img.shape[1]
        box_percentage = (box_area / img_area) * 100

        if max_box_pct < box_percentage:
            max_box_pct = box_percentage

    return round(max_box_pct, 5)

def get_closest_box(all_box_info, box_pct_threshold=1):
    closest_box = None
    closest_distance = float('inf')
    for box_info in all_box_info:
        for each_box in box_info:
            if len(each_box) != 3:
                print(f"box seems empty ! {each_box}")
                continue
            box, box_area, box_percentage = each_box
            if box_percentage > box_pct_threshold:
                box_top_left = (box[0][0], box[0][1])
                distance = (box_top_left[0]**2 + box_top_left[1]**2)**0.5
                if distance < closest_distance:
                    closest_box = each_box[0]
                    closest_distance = distance

    return closest_box


def get_closest_box_y(all_box_info, box_pct_threshold=1):
    closest_box = None
    topmost_box_y = float('inf')
    for box_info in all_box_info:
        for each_box in box_info:
            if len(each_box) != 3:
                print(f"box seems empty ! {each_box}")
                continue
            box, box_area, box_percentage = each_box
            if box_percentage > box_pct_threshold:
                coor_y = box[0][1]
                if box[0][1] < topmost_box_y:
                    topmost_box_y = coor_y
                    closest_box = each_box[0]

    return closest_box


def get_max_box_pct_info(all_box_info, box_pct_threshold=1):
    for box_info in all_box_info:
        for each_box in box_info:
            if len(each_box) != 3:
                print(f"box seems empty ! {each_box}")
                continue
            _, _, box_percentage = each_box
            if box_percentage > box_pct_threshold:
                return True
    return False


def fill_mask(all_box_info, height, width, box_pct_threshold=1):
    mask = np.ones((height, width), dtype=np.uint8) * 255

    for box_info in all_box_info:
        for each_box in box_info:
            if len(each_box) != 3:
                print(f"box seems empty ! {each_box}")
                continue
            box, box_area, box_percentage = each_box
            if box_percentage > box_pct_threshold:
                poly1 = np.array(box).astype(np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [poly1], 0)

    return mask


def calculate_area_pct(box, height, width):
    poly = np.array(box).astype(np.int32).reshape((-1, 2))
    area = cv2.contourArea(poly)
    img_area = height * width
    percentage = (area / img_area) * 100
    return percentage


def saveResult(img_file, img, boxes, closest_box=None, dirname='./result/', verticals=None, texts=None):
    """ save text detection result one by one
    Args:
        img_file (str): image file name
        img (array): raw image context
        boxes (array): array of result file
            Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
    Return:
        None
    """
    img = np.array(img)

    # make result file list
    filename, file_ext = os.path.splitext(os.path.basename(img_file))

    # result directory
    res_file = os.path.join(dirname, f"res_{filename}.txt")
    res_img_file = os.path.join(dirname, f"res_{filename}.jpg")

    if not os.path.isdir(dirname):
        os.mkdir(dirname)

    # box_info = compute_bbox_info(img, boxes) 
    total_percentage = 0
    with open(res_file, 'w') as f:
        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
            ptColor = (0, 255, 255)
            if verticals is not None:
                if verticals[i]:
                    ptColor = (255, 0, 0)

            box_percentage = calculate_area_pct(box, img.shape[0], img.shape[1])
            total_percentage += box_percentage

            # cv2.putText(img, f'Area: {box_info[i][0]:.2f} pixels', (poly[0][0], poly[0][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            cv2.putText(img, f'{box_percentage:.2f}%', (poly[0][0], poly[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            
            # cv2.putText(img, f'Closest bo
            # if closest_box is not None:
            #     cv2.line(img, (0, closest_box[0][1]), (img.shape[1] - 1, closest_box[0][1]), (0, 0, 255), thickness=2)

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(img, "{}".format(texts[i]), (poly[0][0]+1, poly[0][1]+1), font, font_scale, (0, 0, 0), thickness=1)
                cv2.putText(img, "{}".format(texts[i]), tuple(poly[0]), font, font_scale, (0, 255, 255), thickness=1)

    cv2.putText(img, f'{total_percentage:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Save result image
    cv2.imwrite(res_img_file, img)


