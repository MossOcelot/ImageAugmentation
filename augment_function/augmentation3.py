import os
import cv2
import albumentations as A
import numpy as np

# Define paths
image_folder = r'D:\Part-time_job\censor_logoAndface\dataset\test\test_augmentation\image'
label_folder = r'D:\Part-time_job\censor_logoAndface\dataset\test\test_augmentation\labels'
augmented_image_folder = r'D:\Part-time_job\censor_logoAndface\pre_and_post_process\data_augmentation\augment_imagefolder'
augmented_label_folder = r'D:\Part-time_job\censor_logoAndface\pre_and_post_process\data_augmentation\augment_labelfolder'

# Create output folders if they don't exist
os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_label_folder, exist_ok=True)

# Define separate transformations for zoom-out and zoom-in
'''transform_zoom_out = A.Compose([
    A.RandomScale(scale_limit=(-0.4, 0), p=1),  # Zoom-out
    A.PadIfNeeded( border_mode=cv2.BORDER_CONSTANT, value=0)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))'''

'''transform_zoom_in = A.Compose([
    A.RandomScale(scale_limit=(0, 0.3), p=1),  # Zoom-in
    A.RandomCrop(height=256, width=256)  # Crop back to original size
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))'''


# Function to load YOLO format bounding boxes
def load_yolo_bboxes(label_path, img_width, img_height):
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, box_width, box_height = map(float, line.strip().split())
            # Convert YOLO format to Pascal VOC format for Albumentations
            x_min = (x_center - box_width / 2) * img_width
            y_min = (y_center - box_height / 2) * img_height
            x_max = (x_center + box_width / 2) * img_width
            y_max = (y_center + box_height / 2) * img_height
            bboxes.append([x_min, y_min, x_max, y_max])
            class_labels.append(int(class_id))
    return bboxes, class_labels

# Function to convert back to YOLO format
def convert_to_yolo_format(bboxes, img_width, img_height):
    yolo_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        box_width = (x_max - x_min) / img_width
        box_height = (y_max - y_min) / img_height
        yolo_bboxes.append([x_center, y_center, box_width, box_height])
    return yolo_bboxes

# Clip bounding boxes to ensure they stay within image dimensions
def clip_bboxes(bboxes, img_width, img_height):
    clipped_bboxes = []
    for bbox in bboxes:
        x_min = max(0, min(bbox[0], img_width))
        y_min = max(0, min(bbox[1], img_height))
        x_max = max(0, min(bbox[2], img_width))
        y_max = max(0, min(bbox[3], img_height))
        clipped_bboxes.append([x_min, y_min, x_max, y_max])
    return clipped_bboxes

# Define manual cutout function with predefined locations and sizes
def apply_cutout(image, cutout_params):
    """
    Apply cutout to specific locations.
    
    Parameters:
        image (np.array): The image to apply cutouts on.
        cutout_params (list of dicts): Each dict should contain:
            - 'x': x-coordinate of the top-left corner of the cutout
            - 'y': y-coordinate of the top-left corner of the cutout
            - 'height': height of the cutout
            - 'width': width of the cutout
            - 'fill_value': fill value for the cutout region (default: 0)
    """
    mask = image.copy()
    for params in cutout_params:
        x = params['x']
        y = params['y']
        height = params['height']
        width = params['width']
        fill_value = params.get('fill_value', 0)  # Default to black fill if not provided

        # Apply the cutout at the specified location
        mask[y:y + height, x:x + width] = fill_value
    return mask

# Define specific cutout locations and sizes
# ตำเเหน่ง (0,0) คือมุมบนซ้าย
# x,y ที่ใช้ใช้เป็นพิกัดมุมบนซ้ายของส่วน ที่ต้องการจะ cutout ส่วน height, width ก็นับออกไปจากจาก x y
cutout_params = [
    {'x': 0, 'y': 0, 'height': 40, 'width': 40, 'fill_value': 0},
    {'x': 120, 'y': 90, 'height': 30, 'width': 30, 'fill_value': 0},
    {'x': 200, 'y': 200, 'height': 50, 'width': 50, 'fill_value': 0},
    # Add more cutouts as needed
]

# Process each image
for image_name in os.listdir(image_folder):
    if image_name.endswith('.jpg') or image_name.endswith('.png'):
        image_path = os.path.join(image_folder, image_name)
        label_path = os.path.join(label_folder, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        # Load and clip bounding boxes
        bboxes, class_labels = load_yolo_bboxes(label_path, width, height)
        bboxes = clip_bboxes(bboxes, width, height)

        ##################################################### zoom in
        transform_zoom_in = A.Compose([
                            A.RandomScale(scale_limit=(0.2, 0.8), p=1),  # Zoom-in
                            A.RandomCrop(height=height, width=width)  # Crop back to original size
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        #################################################### Zoom-out
        transform_zoom_out = A.Compose([
                            A.RandomScale(scale_limit=(-0.8, -0.1), p=1),  # Zoom-out
                            A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # Apply zoom-out transformation
        augmented_zoom_out = transform_zoom_out(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image_zoom_out = augmented_zoom_out['image']
        augmented_bboxes_zoom_out = augmented_zoom_out['bboxes']

        # Save augmented zoom-out image and labels
        augmented_image_zoom_out_path = os.path.join(augmented_image_folder, 'zoom_out_' + image_name)
        cv2.imwrite(augmented_image_zoom_out_path, cv2.cvtColor(augmented_image_zoom_out, cv2.COLOR_RGB2BGR))

        yolo_bboxes_zoom_out = convert_to_yolo_format(augmented_bboxes_zoom_out, width, height)
        augmented_label_zoom_out_path = os.path.join(augmented_label_folder, 'zoom_out_' + image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(augmented_label_zoom_out_path, 'w') as f:
            for class_id, bbox in zip(class_labels, yolo_bboxes_zoom_out):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Apply zoom-in transformation
        augmented_zoom_in = transform_zoom_in(image=image, bboxes=bboxes, class_labels=class_labels)
        augmented_image_zoom_in = augmented_zoom_in['image']
        augmented_bboxes_zoom_in = augmented_zoom_in['bboxes']

        # Save augmented zoom-in image and labels
        augmented_image_zoom_in_path = os.path.join(augmented_image_folder, 'zoom_in_' + image_name)
        cv2.imwrite(augmented_image_zoom_in_path, cv2.cvtColor(augmented_image_zoom_in, cv2.COLOR_RGB2BGR))

        yolo_bboxes_zoom_in = convert_to_yolo_format(augmented_bboxes_zoom_in, width, height)
        augmented_label_zoom_in_path = os.path.join(augmented_label_folder, 'zoom_in_' + image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(augmented_label_zoom_in_path, 'w') as f:
            for class_id, bbox in zip(class_labels, yolo_bboxes_zoom_in):
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        # Apply cutout with specified parameters
        augmented_image_cutout = apply_cutout(image, cutout_params)

        # Save the augmented cutout image
        augmented_image_cutout_path = os.path.join(augmented_image_folder, 'cutout_' + image_name)
        cv2.imwrite(augmented_image_cutout_path, cv2.cvtColor(augmented_image_cutout, cv2.COLOR_RGB2BGR))

        # Save the corresponding labels (you might want to adjust the labels based on the cutouts)
        # For now, we're just copying the existing labels.
        yolo_bboxes_cutout = convert_to_yolo_format(bboxes, width, height)
        augmented_label_cutout_path = os.path.join(augmented_label_folder, 'cutout_' + image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        with open(augmented_label_cutout_path, 'w') as f:
            for class_id, bbox in zip(class_labels, yolo_bboxes_cutout):  # Note: You may need to adjust bounding boxes if they overlap with cutouts
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")