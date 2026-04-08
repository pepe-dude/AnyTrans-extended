import datetime
import os
import cv2
import numpy as np
from PIL import Image
import math


def image_crop(pil_imgae,boxes,image_path,idx):
    pil_image=cv2.imread(image_path)
    min_x = max(min([x for x, _ in boxes]),0)
    max_x = max([x for x, _ in boxes])
    min_y = max(min([y for _, y in boxes]),0)
    max_y = max([y for _, y in boxes])
    patch_image = pil_image[min_y:max_y, min_x:max_x]
    if image_path[-10:-4]=='reedit':
        save_image_path= image_path[:-11]+f'{idx}crop.png'
    else:
        save_image_path= image_path[:-4]+f'{idx}crop.png'
    cv2.imwrite(save_image_path,patch_image)
    return save_image_path


def resize_image_boxes(img,boxes, max_length=768):
    height, width = img.shape[:2]
    height_original, width_original = height, width
    max_dimension = max(height, width)

    # resize the image if it exceeds the maximum size
    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)

    # Force dimensions devisible by 64
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    height_end, width_end = img.shape[:2]
    height_scale = height_end / height_original
    width_scale = width_end / width_original

    # resize all text boxes accordingly
    new_boxes=[]
    for box_coordinates in boxes:
        box_coordinates = np.array(box_coordinates, dtype=np.int32)
        rect = cv2.minAreaRect(box_coordinates)
        center, size, angle = rect

        # scale the rectangle size
        if angle < 45:
            size_new = (size[0] * width_scale, size[1] * height_scale)    
        else:
            size_new = (size[0] * height_scale, size[1] * width_scale) 

        # align the textbox with the resized image and convert back to 4 corner points
        new_center = (center[0] * width_scale, center[1] * height_scale)          
        new_rect = (new_center, size_new, angle)
        new_box_coordinates = (cv2.boxPoints(new_rect).tolist())

        # sligtly expand the bounding box
        new_coords = []
        for i, coord in enumerate(new_box_coordinates):
            if i in (0, 1):
                new_coords.append([math.floor(x) for x in coord])
            elif i in (2, 3):
                new_coords.append([math.ceil(x) for x in coord])
        new_boxes.append(new_coords)

    return img,new_boxes


def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def enlarge_box_bigger(box):
    width = distance(box[0],box[1])
    height = distance(box[2],box[3])  

    delta_width = width * 0.06
    delta_height = height * 0.06

    # convert into a rectangle
    box = np.array(box, dtype=np.int32)
    rect = cv2.minAreaRect(box)
    center, size, angle = rect

    # exapnd the rectangle
    if angle < 45:
        size_new = (size[0] + delta_width, size[1] + delta_height)
    else:
        size_new = (size[0] + delta_height, size[1] + delta_width)

    # reconstruct the new rectangle into a box
    new_rect = (center, size_new, angle)
    new_box = cv2.boxPoints(new_rect)
    
    return new_box


def resize_mask_returnbox(img_path, box_coordinates, char_count_old, char_count_new, min_scale = 0.7, max_scale = 1.4):
    # Load image and text box
    pil_image=cv2.imread(img_path)
    height,width = pil_image.shape[:2]
    box_coordinates = np.array(box_coordinates, dtype=np.int32)

    # Calculate scale factor and avoid division by zero
    char_count_old = max(char_count_old, 1)
    char_count_new = max(char_count_new, 1)
    scale_factor = char_count_new / char_count_old
    scale_factor = np.clip(scale_factor, min_scale, max_scale)

    center = box_coordinates.mean(axis=0)
    new_box = center + (box_coordinates - center) * scale_factor

    new_box[:, 0] = np.clip(new_box[:, 0], 0, width - 1)
    new_box[:, 1] = np.clip(new_box[:, 1], 0, height - 1) 

    new_box_int = new_box.astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(mask, [new_box_int], 255)
    mask = 255 - mask

    return mask, new_box