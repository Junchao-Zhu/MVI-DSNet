import os
import numpy as np
from PIL import Image, ImageEnhance
import random


def load_image(image_path):
    return Image.open(image_path)


def save_image(image, path, format):
    image.save(path, format=format)


def random_transform(image, annotation):
    # Random rotation
    angle = random.choice([0, 90, 180, 270])
    image = image.rotate(angle)
    annotation = annotation.rotate(angle)

    # Random flipping
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        annotation = annotation.transpose(Image.FLIP_LEFT_RIGHT)

    # Randomly decide whether to adjust brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(random.uniform(0.8, 1.2))

    # Randomly decide whether to add noise
    if random.random() > 0.5:
        noise = np.random.normal(0, 40, (image.height, image.width, 3))
        image = np.array(image) + noise
        image = np.clip(image, 0, 255).astype('uint8')
        image = Image.fromarray(image)

    return image, annotation


def sufficient_annotation(annotation, threshold=0.05):
    # Calculate the proportion of the annotated area to the total area
    anno_array = np.array(annotation)
    target_pixels = np.sum(anno_array > 0)  # Assume target area pixels are greater than 0
    total_pixels = anno_array.size
    return (target_pixels / total_pixels) >= threshold


def process_images(image_dir, annotation_dir, output_dir):
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(image_dir, filename)
            annotation_path = os.path.join(annotation_dir, filename.replace('.jpg', '.png'))  # assuming same filename but different extension

            image = load_image(image_path)
            annotation = load_image(annotation_path)

            # Randomly select 1 to 2 regions from the original image for cropping
            num_crops = random.randint(1, 2)  # Randomly choose to crop 1 or 2 times
            for _ in range(num_crops):
                # Randomly select cropping starting point
                i = random.randint(0, max(0, image.width - 640))
                j = random.randint(0, max(0, image.height - 640))

                img_crop = image.crop((i, j, i + 640, j + 640))
                anno_crop = annotation.crop((i, j, i + 640, j + 640))

                # Apply transformations
                img_transformed, anno_transformed = random_transform(img_crop, anno_crop)

                # Check if annotation is sufficient
                if sufficient_annotation(anno_transformed):
                    # Save image and annotation
                    save_image(
                        img_transformed,
                        os.path.join(output_dir, 'image', f'{filename[:-4]}_{i}_{j}.jpg'),
                        'JPEG'
                    )
                    save_image(
                        anno_transformed,
                        os.path.join(output_dir, 'labels', f'{filename[:-4]}_{i}_{j}_anno.png'),
                        'PNG'
                    )


# Usage example
file_path = r'.\data\train\image'
anno_path = r'.data\train\mask'

process_images(file_path, anno_path, r'.\train')
