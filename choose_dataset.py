import os
import shutil

path = r'D:\Projects\CensorProjectAI\SperateDataset'
obj_class = 4

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


if __name__ == "__main__":
    image_path = rf'{path}\{obj_class}\images'
    image_files = os.listdir(image_path)

    delete_all_files_in_folder(r'dataset\images')
    delete_all_files_in_folder(r'dataset\labels')

    for file_name in image_files:
        name, _ = os.path.splitext(file_name)

        txt_path = rf'{path}\{obj_class}\labels'
        print(txt_path)
        shutil.copy2(rf'{image_path}\{file_name}', rf'dataset\images\{file_name}')
        shutil.copy(rf'{txt_path}\{name}.txt', rf'dataset\labels\{name}.txt')