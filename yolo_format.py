"""
code for yolo format
"""
import os
import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    return np.array(Image.open(image_path))


def find_bounding_boxes(annotation):
    contours, _ = cv2.findContours(annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes


def convert_to_yolo_format(bbox, img_width, img_height):
    x, y, w, h = bbox
    center_x = x + w / 2
    center_y = y + h / 2
    return (center_x / img_width, center_y / img_height, w / img_width, h / img_height)


def draw_boxes(image, bounding_boxes):
    for x, y, w, h in bounding_boxes:
        start_point = (x, y)
        end_point = (x + w, y + h)
        color = (255, 0, 255)  # Red color
        thickness = 5
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image


def process_images(image_dir, annotation_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            anno_file = filename[:-4] + '.png'
            annotation_path = os.path.join(annotation_dir, anno_file)

            image = load_image(image_path)
            annotation = load_image(annotation_path)

            bounding_boxes = find_bounding_boxes(annotation)
            yolo_bboxes = [convert_to_yolo_format(bbox, image.shape[1], image.shape[0]) for bbox in bounding_boxes]

            # Draw and save the images with bounding boxes
            image_with_boxes = draw_boxes(image.copy(), bounding_boxes)
            annotation_with_boxes = draw_boxes(annotation.copy(), bounding_boxes)

            cv2.imwrite(os.path.join(output_dir, 'image', f'{filename[:-4]}_boxed.jpg'), image_with_boxes)
            cv2.imwrite(os.path.join(output_dir, 'label', f'{filename[:-4]}_anno_boxed.png'), annotation_with_boxes)

            # Optionally save YOLO formatted bounding boxes to a file
            with open(os.path.join(output_dir, 'txt', f'{filename[:-4]}.txt'), 'w') as file:
                for bbox in yolo_bboxes:
                    file.write(f'0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n')


# Example usage
process_images(r'.\new_data\test\img', r'.\new_data\test\label', r'.\new_data\yolo_annotation\test')