import numpy as np
import pandas as pd
import os


def parse_detection_file(file_path):
    """Parse a single detection result file and return a list of target data, with results converted back to original image size."""
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 6:  # Ensure the format matches expectations
                # Normalize coordinates and size by multiplying by 640 to restore to original size
                x = int(float(parts[1]) * 640)
                y = int(float(parts[2]) * 640)
                width = int(float(parts[3]) * 640)
                height = int(float(parts[4]) * 640)
                detections.append([x, y, width, height])
    return detections


def load_detections(folder_path):
    """Traverse the folder, parse all files, and aggregate detection results into a large NumPy array."""
    all_detections = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            detections = parse_detection_file(file_path)
            all_detections.extend(detections)

    # Convert the list to a NumPy array with dtype=object (since it contains integers)
    if all_detections:
        return np.array(all_detections, dtype=object)
    else:
        return np.array([])  # Return an empty array if there's no data


def analyze_detection_results(file_name, num_patch, detection_results):
    """Analyze detection results, counting the number of targets, size, size variability, aspect ratio, and other information."""
    if detection_results is None:
        return None

    # Get the total number of objects
    total_objects = len(detection_results)

    # Calculate the area and bounding box width and height
    areas = detection_results[:, 2] * detection_results[:, 3]
    widths = detection_results[:, 2]
    heights = detection_results[:, 3]

    # Calculate the 25th, 50th, and 75th percentiles
    median_area = np.median(areas)
    median_width = np.median(widths)
    median_height = np.median(heights)

    mean_area = np.mean(areas)
    mean_width = np.mean(widths)
    mean_heights = np.mean(heights)

    # Calculate standard deviation
    std_area = np.std(areas)
    std_width = np.std(widths)
    std_height = np.std(heights)

    # Calculate aspect ratios
    aspect_ratios = widths / heights
    mean_aspect_ratio = np.mean(aspect_ratios)
    median_aspect_ratios = np.median(aspect_ratios)
    std_aspect_ratio = np.std(aspect_ratios)

    # Calculate variability coefficients
    coefficient_of_variation_area = std_area / np.mean(areas)
    coefficient_of_variation_width = std_width / np.mean(widths)
    coefficient_of_variation_height = std_height / np.mean(heights)
    coefficient_of_variation_aspect_ratios = std_aspect_ratio / np.mean(aspect_ratios)

    frequency = total_objects / num_patch
    density = np.sum(areas) / (num_patch * 640 * 640)

    # Return statistical information
    extracted_information = {
        'name': file_name[:12],
        'total_objects': total_objects,
        'mean_area': mean_area,
        'mean_width': mean_width,
        'mean_height': mean_heights,
        'mean_aspect_ratio': mean_aspect_ratio,
        'median_area': median_area,
        'median_width': median_width,
        'median_height': median_height,
        'median_aspect_ratio': median_aspect_ratios,
        'std_area': std_area,
        'std_width': std_width,
        'std_height': std_height,
        'std_aspect_ratio': std_aspect_ratio,
        'coefficient_of_variation_area': coefficient_of_variation_area,
        'coefficient_of_variation_width': coefficient_of_variation_width,
        'coefficient_of_variation_height': coefficient_of_variation_height,
        'coefficient_of_variation_aspect_ratios': coefficient_of_variation_aspect_ratios,
        'frequency': frequency,
        'density': density
    }

    df = pd.DataFrame.from_dict(extracted_information, orient='index', columns=['Value'])
    df = df.transpose()

    # Save to Excel file
    excel_file_path = './ours/tcga_information/excels/{}.xlsx'.format(file_name)
    df.to_excel(excel_file_path, index=False)

    print("Statistics saved to", excel_file_path)
