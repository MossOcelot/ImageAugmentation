import albumentations as A
import cv2

def zoom_in(scale_limit:tuple, height, width, p=1):
    transform = A.Compose([
        A.RandomScale(scale_limit=scale_limit, p=p),  # Zoom-in
        A.RandomCrop(height=height, width=width)  # Crop back to original size
    ], bbox_params = A.BboxParams(format="yolo" , label_fields=['labels']))

    return transform

def zoom_out(scale_limit:tuple, height, width, p=1):
    transform = A.Compose([
        A.RandomScale(scale_limit=scale_limit, p=p),  # Zoom-in
        A.PadIfNeeded(min_height=height, min_width=width, border_mode=cv2.BORDER_CONSTANT, value=0)
    ], bbox_params = A.BboxParams(format="yolo" , label_fields=['labels']))

    return transform