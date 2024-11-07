import os
import cv2
import numpy as np
import albumentations as A
from augment_function.cleandataset import read_bbox_txt, read_image, draw_yolo_bboxes
import time

def delete_all_files_in_folder(folder_path):
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # ตรวจสอบว่าเป็นไฟล์และลบ
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Deleted all files in {folder_path}")
    else:
        print(f"The folder {folder_path} does not exist.")

delete_all_files_in_folder(r'augment_result\images')
delete_all_files_in_folder(r'augment_result\labels')

time.sleep(1)

if __name__ == "__main__":
    image_path = r'dataset\images'
    txt_path = r'dataset\labels'

    for file_name in os.listdir(image_path):
        image, width, height = read_image(rf'{image_path}\{file_name}')
        name, _ = os.path.splitext(file_name)
        labels, bboxes = read_bbox_txt(rf'{txt_path}\{name}.txt', width, height)
        
        transform = A.Compose([
            A.Downscale(scale_range=(0.25, 0.35), p=1),
            A.Rotate((-45, 45), border_mode=cv2.BORDER_CONSTANT, p=0.5)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=['labels']))

        try:
            transformed = transform(image=image, bboxes=bboxes, labels=labels)
        except:
            continue

        image = transformed['image']
        labels = transformed['labels']
        bboxes = transformed['bboxes']

        # บันทึกภาพในกรณีที่ไม่พบ bounding box ผิดปกติ
        cv2.imwrite(f'augment_result/images/augment_{file_name}', image)

        with open(f'augment_result/labels/augment_{name}.txt', 'w') as f:
            for i in range(len(labels)):
                f.write(f"{int(labels[i])} {' '.join(map(str, bboxes[i]))}\n")
