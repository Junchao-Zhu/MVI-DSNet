import os
import subprocess


def detect_images_in_folder(folder_name, folder_path, weights, output_dir, img_size=640, conf_thres=0.298, iou_thres=0.45):
    """
    Detects images in a specified folder using YOLOv5 detect.py.

    Args:
    folder_path (str): Path to the folder containing images.
    weights (str): Path to the YOLOv5 weights file.
    output_dir (str): Path to the output directory where results will be saved.
    img_size (int, optional): Inference size (pixels).
    conf_thres (float, optional): Confidence threshold.
    iou_thres (float, optional): IOU threshold for NMS.
    """
    command = [
        'python', 'detect.py',
        '--weights', weights,
        '--source', folder_path,
        '--img-size', str(img_size),
        '--conf-thres', str(conf_thres),
        '--iou-thres', str(iou_thres),
        '--save-txt',  # to save results as .txt files
        '--save-conf', # to save confidence scores
        '--project', output_dir, # specify the output directory
        '--name', folder_name,  # specify the output directory
        '--device', '0'  # overwrite existing output
    ]
    subprocess.run(command)


folders = os.listdir(r'.\yolo_annotation\images')
weights_path = r'.\MVI_detect\models\MVI-Detect.pt'
output_directory = r'.\test_detect'

for folder in folders:
    f_path = os.path.join(r'.\yolo_annotation\images', folder)
    detect_images_in_folder(folder, f_path, weights_path, output_directory)
