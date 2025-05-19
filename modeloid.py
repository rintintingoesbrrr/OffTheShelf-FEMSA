import os
import cv2
import numpy as np
from ultralytics import YOLO
import shutil
import glob
from PIL import Image, ImageDraw, ImageFont
import argparse

def identify_products(crops_folder, output_folder, models_dir="./models", product_model_name="DIOS.pt"):
    """
    Takes a folder of cropped product/empty space images and:
    1. Runs YOLOv8 product classification on the product crops
    2. Creates a composite image with all crops in order with their labels
    3. Saves results in the specified output folder
    
    Args:
        crops_folder: Path to folder containing cropped images
        output_folder: Path to save the results
        models_dir: Directory containing the model files
        product_model_name: Filename of the YOLOv8 product classification model
    
    Returns:
        dict: Contains classification results and paths to output files
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load YOLOv8 model for product classification
    product_model_path = os.path.join(models_dir, product_model_name)
    try:
        model = YOLO(product_model_path)
        print(f"Loaded product classification model from {product_model_path}")
    except Exception as e:
        return {"error": f"Error loading YOLOv8 model: {str(e)}"}
    
    # Get all crop image files
    all_files = glob.glob(os.path.join(crops_folder, "*.jpg")) + \
                glob.glob(os.path.join(crops_folder, "*.jpeg")) + \
                glob.glob(os.path.join(crops_folder, "*.png"))
    
    if not all_files:
        return {"error": f"No image files found in {crops_folder}"}
    
    # Sort files by their IDs to maintain order
    # Extract row and column numbers from filenames (format is like "P12_uuid_filename.jpg" or "E12_uuid_filename.jpg")
    def extract_id_info(filename):
        basename = os.path.basename(filename)
        # Get the type (P or E), row (1st digit after P/E), and column (2nd digit after P/E)
        type_char = basename[0]  # P or E
        row = int(basename[1])
        col = int(basename[2])
        return (row, col, type_char, filename)
    
    all_files_sorted = sorted(all_files, key=extract_id_info)
    
    # Process each crop and build results
    classification_results = []
    product_images = []
    
    for file_path in all_files_sorted:
        basename = os.path.basename(file_path)
        
        # Check if this is a product or empty space
        is_product = basename.startswith('P')
        
        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Could not read image {file_path}. Skipping.")
            continue
        
        # For products, run product classification/detection
        if is_product:
            # Run inference to classify/detect product
            results = model(image)
            
            # Extract prediction
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                # Check if this is a classification model
                if hasattr(result, 'probs') and result.probs is not None:
                    # Classification model
                    class_idx = int(result.probs.top1)
                    confidence = float(result.probs.top1conf)
                    predicted_class = result.names[class_idx]
                else:
                    # Detection model
                    if len(result.boxes) > 0:
                        # Get the box with highest confidence
                        best_box = max(result.boxes, key=lambda x: float(x.conf[0]))
                        class_idx = int(best_box.cls[0])
                        confidence = float(best_box.conf[0])
                        predicted_class = result.names[class_idx]
                    else:
                        predicted_class = "unknown"
                        confidence = 0.0
            else:
                predicted_class = "unknown"
                confidence = 0.0
            
            # Create label with classification info
            label = f"{basename[0:3]}: {predicted_class} ({confidence:.2f})"
        else:
            # For empty spaces, just use the ID as label
            label = f"{basename[0:3]}: Empty"
            predicted_class = "empty"
            confidence = 1.0
        
        # Save classification info
        classification_results.append({
            "id": basename[0:3],  # Get PRC or ERC ID
            "type": "product" if is_product else "empty_space",
            "file_path": file_path,
            "predicted_class": predicted_class,
            "confidence": confidence
        })
        
        # Add label to image for the composite
        height, width, _ = image.shape
        # Convert from BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Create a slightly larger image to accommodate the label
        label_height = 30
        new_img = Image.new('RGB', (width, height + label_height), color=(255, 255, 255))
        new_img.paste(pil_image, (0, 0))
        
        # Add label text
        draw = ImageDraw.Draw(new_img)
        
        # Try to get a font, use default if not available
        try:
            # Try to use a default font (adjust path for your system)
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            # If no truetype font is available, use default
            font = ImageFont.load_default()
        
        # Set label color based on product or empty space
        label_color = (0, 155, 0) if is_product else (155, 0, 0)
        draw.text((5, height + 5), label, fill=label_color, font=font)
        
        product_images.append(new_img)
    
    # Create the composite image
    # First, determine the layout
    num_images = len(product_images)
    
    if num_images == 0:
        return {"error": "No valid images found to process"}
    
    # Determine how many images per row in the composite
    max_images_per_row = 5  # Adjust as needed
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
    
    # Calculate the size of the composite image
    if product_images:
        img_width = product_images[0].width
        img_height = product_images[0].height
    else:
        img_width = 200
        img_height = 200
    
    # Create the composite image
    composite_width = min(num_images, max_images_per_row) * img_width
    composite_height = num_rows * img_height
    
    composite_image = Image.new('RGB', (composite_width, composite_height), (255, 255, 255))
    
    # Place images in the composite
    for i, img in enumerate(product_images):
        row = i // max_images_per_row
        col = i % max_images_per_row
        x = col * img_width
        y = row * img_height
        composite_image.paste(img, (x, y))
    
    # Save the composite image
    composite_image_path = os.path.join(output_folder, f"composite_{os.path.basename(crops_folder)}.jpg")
    composite_image.save(composite_image_path)
    
    # Create a JSON-serializable result
    result = {
        "classifications": classification_results,
        "composite_image_path": composite_image_path,
        "num_products": sum(1 for r in classification_results if r["type"] == "product"),
        "num_empty_spaces": sum(1 for r in classification_results if r["type"] == "empty_space")
    }
    
    # Save classification results to a text file for easy review
    results_file_path = os.path.join(output_folder, f"results_{os.path.basename(crops_folder)}.txt")
    
    with open(results_file_path, 'w') as f:
        f.write(f"Classification Results for {crops_folder}\n")
        f.write(f"Total items: {len(classification_results)}\n")
        f.write(f"Products: {result['num_products']}\n")
        f.write(f"Empty spaces: {result['num_empty_spaces']}\n\n")
        
        for item in classification_results:
            f.write(f"ID: {item['id']}, Type: {item['type']}\n")
            if item['type'] == 'product':
                f.write(f"  Predicted class: {item['predicted_class']}, Confidence: {item['confidence']:.2f}\n")
        
    result["results_file_path"] = results_file_path
    
    print(f"Processed {len(classification_results)} items ({result['num_products']} products, {result['num_empty_spaces']} empty spaces)")
    print(f"Composite image saved to: {composite_image_path}")
    print(f"Results saved to: {results_file_path}")
    
    return result
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process cropped product images with YOLOv8')
    parser.add_argument('--crops_folder', required=True, help='Path to folder containing cropped images')
    parser.add_argument('--output_folder', required=True, help='Path to save the output')
    parser.add_argument('--models_dir', default='./models', help='Directory containing model files')
    parser.add_argument('--product_model', default='DIOS.pt', help='Name of product classification model')
    
    args = parser.parse_args()
    
    # Call the main function
    result = identify_products(
        args.crops_folder, 
        args.output_folder, 
        args.models_dir, 
        args.product_model
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()