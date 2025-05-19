import cv2
import torch
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

def analyze_shelf_products(image_data, models_dir="./models"):
    """
    Analyzes shelf products using both YOLOv5 for product detection
    and YOLOv8 for empty space detection.
    
    Args:
        image_data: Can be a file path (str) or a numpy array containing the image
        models_dir: Directory containing the model files
        
    Returns:
        dict: Contains product arrays, statistics, and the analyzed image
    """
    # Check if image_data is a file path or numpy array
    if isinstance(image_data, str):
        image = cv2.imread(image_data)
        if image is None:
            return {"error": f"Error: Could not read image from {image_data}"}
    else:
        image = image_data  # Assume it's already a numpy array
    
    # Load YOLOv5 model for product detection
    yolov5_model_path = f"{models_dir}/jonathan.pt"
    try:
        model_v5 = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_model_path)
    except Exception as e:
        return {"error": f"Error loading YOLOv5 model: {str(e)}"}
    
    # Load YOLOv8 model for empty space detection
    yolov8_model_path = f"{models_dir}/empty_spaces.pt"
    try:
        model_v8 = YOLO(yolov8_model_path)
    except Exception as e:
        return {"error": f"Error loading YOLOv8 model: {str(e)}"}
    
    # Set confidence thresholds
    model_v5.conf = 0.25  # Confidence threshold for YOLOv5
    model_v5.iou = 0.45   # IoU threshold for YOLOv5
    
    # Make predictions with YOLOv8 (empty space detection) FIRST
    results_v8 = model_v8(image)
    
    # Make predictions with YOLOv5 (product detection) SECOND
    results_v5 = model_v5(image)
    
    # Extract product detections from YOLOv5
    predictions_v5 = results_v5.xyxy[0].cpu().numpy()  # Get detections in xyxy format
    
    # Filter for only "product" class from YOLOv5
    product_detections = []
    for det in predictions_v5:
        x1, y1, x2, y2, conf, cls = det
        class_idx = int(cls)
        class_name = results_v5.names[class_idx]
        if class_name.lower() == "product":
            # Store as (x1, y1, x2, y2, conf)
            product_detections.append((int(x1), int(y1), int(x2), int(y2), conf))
    
    # Extract empty space detections from YOLOv8
    empty_space_detections = []
    # Process YOLOv8 results
    for result in results_v8:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            class_name = model_v8.names[int(cls)]
            if class_name.lower() in ["empty", "gap", "empty_space"]:
                # Store as (x1, y1, x2, y2, conf)
                empty_space_detections.append((int(x1), int(y1), int(x2), int(y2), conf))
    
    # Improved empty space detection by filling gaps between products
    # Group products into rows based on y-coordinate
    y_threshold = 50  # Adjust this threshold based on your images
    
    # Sort products by y-coordinate (top to bottom)
    product_detections.sort(key=lambda x: (x[1] + x[3]) / 2)
    
    # Group by rows
    rows = []
    current_row = []
    last_y_center = None
    for detection in product_detections:
        x1, y1, x2, y2, conf = detection
        y_center = (y1 + y2) / 2
        if last_y_center is None or abs(y_center - last_y_center) <= y_threshold:
            current_row.append(detection)
        else:
            if current_row:  # Only add non-empty rows
                rows.append(current_row)
            current_row = [detection]
        last_y_center = y_center
    
    # Add the last row if not empty
    if current_row:
        rows.append(current_row)
    
    # Also group empty spaces into the same row structure
    empty_space_rows = [[] for _ in range(len(rows))]
    for empty_space in empty_space_detections:
        x1, y1, x2, y2, conf = empty_space
        empty_space_center_y = (y1 + y2) / 2
        
        # Find which row this empty space belongs to
        assigned = False
        for row_idx, row in enumerate(rows):
            if row:  # Only check non-empty rows
                row_center_y = sum((p[1] + p[3]) / 2 for p in row) / len(row)
                if abs(empty_space_center_y - row_center_y) <= y_threshold:
                    empty_space_rows[row_idx].append(empty_space)
                    assigned = True
                    break
        
        # If it doesn't fit into any existing row, check if it should form a new row
        if not assigned:
            if rows:
                # Get all row centers
                row_centers = [sum((p[1] + p[3]) / 2 for p in row) / len(row) if row else 0 for row in rows]
                row_centers = [y for y in row_centers if y > 0]  # Filter out empty rows
                row_centers.sort()
                
                # Check if this empty space could be a new row between existing rows
                for i in range(len(row_centers) - 1):
                    mid_point = (row_centers[i] + row_centers[i+1]) / 2
                    if abs(empty_space_center_y - mid_point) <= y_threshold * 0.8:  # Slightly stricter threshold
                        # Create a new row for this empty space
                        insert_idx = i + 1
                        rows.insert(insert_idx, [])
                        empty_space_rows.insert(insert_idx, [empty_space])
                        assigned = True
                        break
                
                # Check if it could be a new row above the first row or below the last row
                if not assigned:
                    if empty_space_center_y < row_centers[0] - y_threshold * 0.8:
                        # New row above the first row
                        rows.insert(0, [])
                        empty_space_rows.insert(0, [empty_space])
                    elif empty_space_center_y > row_centers[-1] + y_threshold * 0.8:
                        # New row below the last row
                        rows.append([])
                        empty_space_rows.append([empty_space])
            else:
                # If there are no product rows yet, create a row for this empty space
                rows.append([])
                empty_space_rows.append([empty_space])
    
    # IMPROVED: Infer empty spaces between products in the same row
    for row_idx, (row_products, row_empty_spaces) in enumerate(zip(rows, empty_space_rows)):
        if len(row_products) >= 2:
            # Sort products from left to right
            row_products.sort(key=lambda x: x[0])
            
            # Detect gaps between adjacent products
            for i in range(len(row_products) - 1):
                curr_product = row_products[i]
                next_product = row_products[i + 1]
                curr_x1, curr_y1, curr_x2, curr_y2, curr_conf = curr_product
                next_x1, next_y1, next_x2, next_y2, next_conf = next_product
                
                # Check if there's a significant gap between these products
                gap_size = next_x1 - curr_x2
                min_gap_size = 30  # Minimum gap size to consider it an empty space (adjust as needed)
                
                if gap_size > min_gap_size:
                    # Calculate the average height of the two products
                    avg_height = ((curr_y2 - curr_y1) + (next_y2 - next_y1)) / 2
                    
                    # Create a new empty space detection between these products
                    gap_x1 = curr_x2
                    gap_x2 = next_x1
                    gap_y1 = max(curr_y1, next_y1)
                    gap_y2 = min(curr_y2, next_y2)
                    
                    # If the heights are very different, adjust to make a reasonable box
                    if gap_y2 - gap_y1 < avg_height * 0.5:
                        mid_y = (gap_y1 + gap_y2) / 2
                        gap_y1 = int(mid_y - avg_height / 2)
                        gap_y2 = int(mid_y + avg_height / 2)
                    
                    # Use a reasonable confidence value for inferred gaps (lower than detected ones)
                    inferred_conf = 0.7
                    
                    # Check if this inferred gap overlaps with any detected empty space
                    overlaps = False
                    for ex1, ey1, ex2, ey2, econf in row_empty_spaces:
                        # Check for overlap
                        if (gap_x1 < ex2 and gap_x2 > ex1 and 
                            gap_y1 < ey2 and gap_y2 > ey1):
                            overlaps = True
                            break
                    
                    if not overlaps:
                        # Add this inferred empty space
                        empty_space = (int(gap_x1), int(gap_y1), int(gap_x2), int(gap_y2), inferred_conf)
                        row_empty_spaces.append(empty_space)
    
    # Process each row to identify frontmost products and gaps
    frontmost_products = []
    empty_spaces = []
    product_arrays = []
    
    for row_idx, (row, empty_row) in enumerate(zip(rows, empty_space_rows)):
        # Sort the row from left to right by x coordinate
        row.sort(key=lambda x: x[0])
        empty_row.sort(key=lambda x: x[0])
        
        # For each row, we'll create a list of frontmost products
        row_frontmost = []
        row_empty_spaces = empty_row.copy()  # Start with all detected empty spaces
        occupied_segments = []
        
        # Process each product in the row
        for product in row:
            x1, y1, x2, y2, conf = product
            product_width = x2 - x1
            product_center_x = (x1 + x2) / 2
            
            # Check if this product overlaps significantly with any existing frontmost product
            is_behind = False
            for seg_start, seg_end in occupied_segments:
                # If the center of the current product falls within an occupied segment
                if product_center_x >= seg_start and product_center_x <= seg_end:
                    is_behind = True
                    break
            
            # If it's not behind any existing product, it's a frontmost product
            if not is_behind:
                row_frontmost.append(product)
                # Add its horizontal span to occupied segments
                segment_width = product_width * 0.8  # 80% of actual width
                segment_start = max(0, product_center_x - segment_width/2)
                segment_end = product_center_x + segment_width/2
                occupied_segments.append((segment_start, segment_end))
        
        # Sort frontmost products from left to right
        row_frontmost.sort(key=lambda x: x[0])
        frontmost_products.append(row_frontmost)
        empty_spaces.append(row_empty_spaces)
        
        # Create array for this row, integrating product and empty space detection
        if row_frontmost or row_empty_spaces:
            # Determine the average width of an item (product or gap) in this row
            item_widths = []
            for x1, _, x2, _, _ in row_frontmost:
                item_widths.append(x2 - x1)
            for x1, _, x2, _, _ in row_empty_spaces:
                item_widths.append(x2 - x1)
            
            avg_width = sum(item_widths) / len(item_widths) if item_widths else 100  # Default if no items
            
            # Combine product and empty space detections
            combined_items = []
            for product in row_frontmost:
                x1, y1, x2, y2, conf = product
                combined_items.append((x1, y1, x2, y2, conf, 1))  # 1 indicates product
            
            for empty in row_empty_spaces:
                x1, y1, x2, y2, conf = empty
                combined_items.append((x1, y1, x2, y2, conf, 0))  # 0 indicates empty space
            
            # Sort all items from left to right
            combined_items.sort(key=lambda x: x[0])
            
            # Handle overlapping detections and create the final array
            row_array = []
            last_end_x = 0
            
            for x1, y1, x2, y2, conf, item_type in combined_items:
                # Check if there's a significant gap before this item
                if x1 - last_end_x > avg_width * 0.7:
                    # Estimate number of missing items
                    gap_size = x1 - last_end_x
                    estimated_missing = max(0, int(gap_size / avg_width + 0.5) - 1)
                    # Add placeholders based on what we expect but didn't detect
                    # We'll mark these as "?" in visualization later
                    row_array.extend([2] * estimated_missing)  # 2 indicates unknown/undetected
                
                # Add the current item
                row_array.append(item_type)
                last_end_x = max(last_end_x, x2)
            
            # If this row had no detections, add a placeholder
            if not row_array:
                row_array = []
        else:
            # Empty row
            row_array = []
        
        product_arrays.append(row_array)
    
    # Render results on the image for visualization
    rendered_image = image.copy()
    
    # Draw all detected products and empty spaces as faint outlines
    for row in rows:
        for detection in row:
            x1, y1, x2, y2, conf = detection
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(rendered_image, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    for row in empty_space_rows:
        for detection in row:
            x1, y1, x2, y2, conf = detection
            cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (100, 100, 255), 1)
            cv2.putText(rendered_image, f"{conf:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)
    
    # Draw frontmost products and empty spaces with bright colors
    legend_offset = 150  # Starting y-position for the legend
    cv2.putText(rendered_image, "Legend:", (image.shape[1] - 200, legend_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Add legend items
    cv2.rectangle(rendered_image, (image.shape[1] - 200, legend_offset + 20), 
                 (image.shape[1] - 180, legend_offset + 40), (0, 255, 0), -1)
    cv2.putText(rendered_image, "Product", (image.shape[1] - 170, legend_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(rendered_image, (image.shape[1] - 200, legend_offset + 50), 
                 (image.shape[1] - 180, legend_offset + 70), (0, 0, 255), -1)
    cv2.putText(rendered_image, "Empty Space", (image.shape[1] - 170, legend_offset + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(rendered_image, (image.shape[1] - 200, legend_offset + 80), 
                 (image.shape[1] - 180, legend_offset + 100), (255, 255, 0), -1)
    cv2.putText(rendered_image, "Unknown", (image.shape[1] - 170, legend_offset + 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    for row_index, (row_products, row_empty, row_array) in enumerate(zip(frontmost_products, empty_spaces, product_arrays)):
        row_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (0, 255, 255), (255, 0, 255)][row_index % 6]
        
        # Draw label for the row
        if row_products or row_empty:
            label_x = 20
            label_y = 150 + row_index * 30
            total_products = sum(1 for x in row_array if x == 1)
            total_gaps = sum(1 for x in row_array if x == 0)
            total_unknown = sum(1 for x in row_array if x == 2)
            
            cv2.putText(rendered_image, f"Row {row_index + 1}: {total_products} products, {total_gaps} gaps, {total_unknown} unknown", 
                        (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, row_color, 2)
            
            # Get combined items (both products and empty spaces) for visualization
            all_items = []
            for item in row_products:
                x1, y1, x2, y2, conf = item
                all_items.append((x1, y1, x2, y2, conf, 1))  # 1 for product
            
            for item in row_empty:
                x1, y1, x2, y2, conf = item
                all_items.append((x1, y1, x2, y2, conf, 0))  # 0 for empty space
            
            # Sort all items from left to right
            all_items.sort(key=lambda x: x[0])
            
            # Draw boxes for each item with appropriate colors
      # Draw boxes for each item with appropriate colors
            for i, (x1, y1, x2, y2, conf, item_type) in enumerate(all_items):
                if item_type == 1:  # Product
                    cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(rendered_image, f"P{i+1} ({conf:.2f})", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:  # Empty space
                    cv2.rectangle(rendered_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(rendered_image, f"E{i+1} ({conf:.2f})", (x1 + 5, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
    # Add summary information
    total_products = sum(sum(1 for x in row if x == 1) for row in product_arrays)
    total_gaps = sum(sum(1 for x in row if x == 0) for row in product_arrays)
    total_unknown = sum(sum(1 for x in row if x == 2) for row in product_arrays)
    
    cv2.putText(rendered_image, f"Total Rows: {len(rows)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(rendered_image, f"Total Products: {total_products}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(rendered_image, f"Total Gaps: {total_gaps}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.putText(rendered_image, f"Total Unknown: {total_unknown}", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Create a summary table at the bottom
    y_offset = image.shape[0] - 30 * len(rows) - 40
    cv2.putText(rendered_image, "Product Array Structure (1=product, 0=gap, 2=unknown):", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    for i, row_array in enumerate(product_arrays):
        row_color = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
                     (0, 255, 255), (255, 0, 255)][i % 6]
        array_str = str(row_array)
        cv2.putText(rendered_image, f"Row {i+1}: {array_str}", 
                    (30, y_offset + 30 * (i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, row_color, 2)
    
    # Create detection coordinates data structure for API response
    detection_data = {
        "rows": []
    }
    
    for row_idx, (row_products, row_empty) in enumerate(zip(frontmost_products, empty_spaces)):
        row_data = {
            "products": [],
            "empty_spaces": []
        }
        
        for p_idx, (x1, y1, x2, y2, conf) in enumerate(row_products):
            row_data["products"].append({
                "id": f"P{p_idx+1}",
                "coordinates": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": float(conf)
            })
        
        for e_idx, (x1, y1, x2, y2, conf) in enumerate(row_empty):
            row_data["empty_spaces"].append({
                "id": f"E{e_idx+1}",
                "coordinates": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                },
                "confidence": float(conf)
            })
                
        detection_data["rows"].append(row_data)
    
    # Return the analysis results for API response
    result = {
        "product_arrays": product_arrays,
        "statistics": {
            "num_rows": len(rows),
            "total_products": total_products,
            "total_gaps": total_gaps,
            "total_unknown": total_unknown
        },
        "detections": detection_data,
        "rendered_image": rendered_image  # This will be encoded to base64 in the API
    }
    
    return result


# Testing function - for local use only, this would not be part of the server
def test_analyzer(image_path, models_dir="./models"):
    """
    Test function for local use. Not used in server deployment.
    """
    result = analyze_shelf_products(image_path, models_dir)
    
    if "error" in result:
        print(result["error"])
        return
    
    # Save the output image
    output_path = image_path.rsplit(".", 1)[0] + "_analyzed.jpg"
    cv2.imwrite(output_path, result["rendered_image"])
    
    # Display the resulting image
    cv2.imshow('Shelf Analysis with Products and Empty Spaces', result["rendered_image"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Print summary info
    print("\nSummary:")
    print(f"Total rows (shelves): {result['statistics']['num_rows']}")
    print(f"Total products: {result['statistics']['total_products']}")
    print(f"Total gaps: {result['statistics']['total_gaps']}")
    print(f"Total unknown: {result['statistics']['total_unknown']}")
    
    print("\nProduct array structure (1=product, 0=gap, 2=unknown):")
    for i, row_array in enumerate(result["product_arrays"]):
        products = sum(1 for x in row_array if x == 1)
        gaps = sum(1 for x in row_array if x == 0)
        unknown = sum(1 for x in row_array if x == 2)
        print(f"Row {i+1}: {row_array} ({products} products, {gaps} gaps, {unknown} unknown)")


if __name__ == "__main__":
    import sys
    import os
    
    # Get image path from command line arguments or use default
    image_path = "./sin_frijol.jpg"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Ensure file exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    # Check if models exist
    models_dir = "./models"
    yolov5_model = f"{models_dir}/jonathan.pt"
    yolov8_model = f"{models_dir}/empty_spaces.pt"
    
    if not os.path.isfile(yolov5_model):
        print(f"Warning: YOLOv5 model not found at {yolov5_model}")
        print("Please ensure you have the product detection model.")
    
    if not os.path.isfile(yolov8_model):
        print(f"Warning: YOLOv8 model not found at {yolov8_model}")
        print("Please ensure you have the empty space detection model.")
    
    # Run the test
    test_analyzer(image_path, models_dir)