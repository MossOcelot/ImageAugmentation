import cv2

def read_image(file_path:str):
    image = cv2.imread(file_path)

    if image is None:
        raise FileNotFoundError(f"Cannot find or open the image at {file_path}")
    
    # ดึงขนาดความกว้างและความสูง
    image_height, image_width = image.shape[:2]
    
    return image, image_width, image_height

def read_bbox_txt(file_path:str, image_width, image_height):
    pixel_bboxes = []
    labels = []

    with open(file_path, 'r') as file:
        for line in file:
            # แปลงข้อมูลในแต่ละบรรทัดเป็น float
            yolo_bbox = list(map(float, line.strip().split()))
            # แปลงจาก YOLO format ไปเป็นพิกเซล format
            class_id, x_center, y_center, width, height = yolo_bbox

            pixel_bboxes.append([x_center, y_center, width, height])
            labels.append(class_id)

    return labels, pixel_bboxes


def yolo_to_pascal_voc(bbox, img_width, img_height):
    """
    แปลง bounding box จาก YOLO format เป็น Pascal VOC format
    Parameters:
    - bbox: [class_id, x_center, y_center, width, height] (normalized coordinates)
    - img_width: ความกว้างของภาพ
    - img_height: ความสูงของภาพ
    """
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    return [x_min, y_min, x_max, y_max]

def draw_yolo_bboxes(image, yolo_bboxes, labels=None, color=(0, 255, 0), thickness=2):
    """
    วาด bounding boxes บนภาพจาก YOLO format
    
    Parameters:
    - image: ภาพต้นฉบับในรูปแบบ NumPy array
    - yolo_bboxes: ลิสต์ของ bounding boxes ใน YOLO format [class_id, x_center, y_center, width, height]
    - labels: ลิสต์ของ label สำหรับ bounding boxes (ถ้ามี)
    - color: สีของกรอบสี่เหลี่ยม (ค่าเริ่มต้นเป็นสีเขียว)
    - thickness: ความหนาของกรอบสี่เหลี่ยม
    """
    img_height, img_width = image.shape[:2]
    
    for i, yolo_bbox in enumerate(yolo_bboxes):
        # แปลง YOLO format เป็น Pascal VOC format
        x_min, y_min, x_max, y_max = yolo_to_pascal_voc(yolo_bbox, img_width, img_height)
        
        # วาดกรอบสี่เหลี่ยมรอบ bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)
        
    return image
