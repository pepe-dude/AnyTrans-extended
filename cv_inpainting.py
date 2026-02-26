import cv2
import os
import numpy as np

def create_mask(pil_image, boxes):
    height,width  = pil_image.shape[:2]
    image_size = (height,width )
    mask = np.zeros(image_size, dtype=np.uint8)

    
    boxes = np.array(boxes, dtype=np.int32)
    cv2.fillPoly(mask, [np.array(boxes, np.int32)], 255)

    return mask

def InpaintImage(imagePath, dt_boxes):
    image = cv2.imread(imagePath)

    mask = create_mask(image, dt_boxes)
    inpaintedImage = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    erased_image_path=imagePath[:-4]+'_erase.png'
    cv2.imwrite(erased_image_path, inpaintedImage)
