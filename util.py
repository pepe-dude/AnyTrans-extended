import os
import datetime
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math


def get_box_dimentions(box):
    height = np.linalg.norm(box[1] - box[0])
    width = np.linalg.norm(box[2] - box[1])
    
    if height > width:
        width, height = height, width

    return width, height


def reorder_box_points(box):
    sum = box.sum(axis=1)
    diff = np.diff(box, axis=1)

    top_left = box[np.argmin(sum)]
    bottom_right = box[np.argmax(sum)]
    top_right = box[np.argmin(diff)]
    bottom_left = box[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left])


def normalize_box(box):
    box = reorder_box_points(box)

    width = np.linalg.norm(box[1] - box[0])
    height = np.linalg.norm(box[3] - box[0])

    if height > width:
        box = np.array([box[1], box[2], box[3], box[0]])

    return box


def get_box_angle(box):
    box = normalize_box(box)

    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]

    angle = np.degrees(np.arctan2(dy, dx))

    return angle


def get_text_size(text, font):
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    return draw.textbbox((0, 0), text, font = font)[2:]


def find_optimal_font_size(text, box_width, box_height, font_path):
    min_size = 5
    max_size = 200
    best_size = min_size

    while min_size <= max_size:
        mid = (min_size + max_size) // 2
        font = ImageFont.truetype(font_path, mid)

        text_w, text_h = get_text_size(text, font)

        if text_w <= box_width and text_h <= box_height:
            best_size = mid
            min_size = mid + 1
        else:
            max_size = mid - 1

    return best_size


def create_text_image(text, font_path, font_size, color=(0,0,0)):
    font = ImageFont.truetype(font_path, font_size)

    # Estimate size
    dummy = Image.new("RGBA", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Create transparent image with padding
    img = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    draw.text((10, 10), text, font=font, fill=color)

    return img


def paste_rotated_text(base_image, text_img, center):
    base = Image.fromarray(base_image).convert("RGBA")

    text_w, text_h = text_img.size

    # top-left position
    x = int(center[0] - text_w / 2)
    y = int(center[1] - text_h / 2)

    base.paste(text_img, (x, y), text_img)

    return np.array(base.convert("RGB"))


def image_crop(boxes, image_path, idx):
    pil_image=cv2.imread(image_path)

    min_x = int(max(min([x for x, _ in boxes]),0))
    max_x = int(max([x for x, _ in boxes]))

    min_y = int(max(min([y for _, y in boxes]),0))
    max_y = int(max([y for _, y in boxes]))

    patch_image = pil_image[min_y:max_y, min_x:max_x]

    save_image_path= image_path[:-4]+f'{idx}crop.png'
        
    cv2.imwrite(save_image_path,patch_image)
    return save_image_path


def save_images(img_list, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    folder_path = os.path.join(folder, date_str)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    time_str = now.strftime("%H_%M_%S")
    for idx, img in enumerate(img_list):
        image_number = idx + 1
        filename = f"{time_str}_{image_number}.jpg"
        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, img[..., ::-1])


def check_channels(image):
    channels = image.shape[2] if len(image.shape) == 3 else 1
    if channels == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif channels > 3:
        image = image[:, :, :3]
    return image


def resize_image(img, max_length=768):
    height, width = img.shape[:2]
    max_dimension = max(height, width)

    if max_dimension > max_length:
        scale_factor = max_length / max_dimension
        new_width = int(round(width * scale_factor))
        new_height = int(round(height * scale_factor))
        new_size = (new_width, new_height)
        img = cv2.resize(img, new_size)
    height, width = img.shape[:2]
    img = cv2.resize(img, (width-(width % 64), height-(height % 64)))
    return img


def resize_image_boxes(img, boxes, max_length=768):
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


def get_contextual_crop(image, box, padding_ratio=1.5):
    """
    Extracts a region around the box with some padding for context.
    Returns the cropped image, the bounding box of the crop in the original image,
    and the box coordinates relative to the crop.
    """
    h, w = image.shape[:2]
    
    # Get box bounds
    min_x = np.min(box[:, 0])
    max_x = np.max(box[:, 0])
    min_y = np.min(box[:, 1])
    max_y = np.max(box[:, 1])
    
    box_w = max_x - min_x
    box_h = max_y - min_y
    
    # Calculate padding
    pad_w = box_w * padding_ratio
    pad_h = box_h * padding_ratio
    
    # Define crop region
    crop_x1 = int(max(0, min_x - pad_w))
    crop_y1 = int(max(0, min_y - pad_h))
    crop_x2 = int(min(w, max_x + pad_w))
    crop_y2 = int(min(h, max_y + pad_h))
    
    crop_img = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # Adjust box coordinates to be relative to the crop
    relative_box = box.copy().astype(np.float32)
    relative_box[:, 0] -= crop_x1
    relative_box[:, 1] -= crop_y1
    
    return crop_img, (crop_x1, crop_y1, crop_x2, crop_y2), relative_box


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


def create_mask(pil_image, boxes):
    height,width  = pil_image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    boxes = np.array(boxes, dtype=np.int32)
    cv2.fillPoly(mask, [np.array(boxes, np.int32)], 255)

    return mask


def inpaint_image(imagePath, dt_boxes):
    image = cv2.imread(imagePath)

    mask = create_mask(image, dt_boxes)
    inpaintedImage = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    cv2.imwrite(imagePath, inpaintedImage)
