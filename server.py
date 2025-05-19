from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os
import uuid
import base64
from werkzeug.utils import secure_filename
from pipeline import analyze_shelf_products
import json
from modeloid import identify_products

import shutil
import traceback

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODELS_DIR = './models'

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set maximum file size to 16MB
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "Server is running"})



@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Endpoint to analyze shelf products in an uploaded image
    """
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Check if the file has a name
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    # Check if the file has an allowed extension
    if not allowed_file(file.filename):
        return jsonify({"error": f"File extension not allowed. Use {', '.join(ALLOWED_EXTENSIONS)}"}), 400
    
    try:
        # Generate a unique filename
        unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Analyze the shelf products
        result = analyze_shelf_products(file_path, MODELS_DIR)
        
        # Check if there was an error in the analysis
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Save the analyzed image
        output_filename = f"analyzed_{unique_filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        cv2.imwrite(output_path, result["rendered_image"])
        
        # Load the original image for cropping
        original_image = cv2.imread(file_path)
        
        # Create a directory for cropped images
        crop_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"crops_{unique_filename.split('.')[0]}")
        os.makedirs(crop_dir, exist_ok=True)
        
        # Process each row and crop detected items
        cropped_images = []
        for row_idx, row_data in enumerate(result["detections"]["rows"]):
            # Combine products and empty spaces for proper labeling
            all_items = []
            
            # Add products with their type
            for prod_idx, product in enumerate(row_data["products"]):
                all_items.append((product, "product"))
            
            # Add empty spaces with their type
            for empty_idx, empty in enumerate(row_data["empty_spaces"]):
                all_items.append((empty, "empty_space"))
            
            # Sort all items from left to right based on x1 coordinate
            all_items.sort(key=lambda item: item[0]["coordinates"]["x1"])
            
            # Now process each item with its correct position index
            for item_idx, (item, item_type) in enumerate(all_items):
                coords = item["coordinates"]
                x1, y1, x2, y2 = coords["x1"], coords["y1"], coords["x2"], coords["y2"]
                
                # Crop the image
                cropped = original_image[y1:y2, x1:x2]
                
                # Skip if cropped area is empty
                if cropped.size == 0:
                    continue
                
                # Generate filename and ID for this crop with the correct position
                # Format as P for product or E for empty space, followed by row and position
                prefix = "P" if item_type == "product" else "E"
                crop_id = f"{prefix}{row_idx+1}{item_idx+1}"
                crop_filename = f"{crop_id}_{unique_filename}"
                crop_path = os.path.join(crop_dir, crop_filename)
                
                # Save the cropped image
                cv2.imwrite(crop_path, cropped)
                
                # Update item ID in the result data
                item["id"] = crop_id
                
                # Add to our list of crops
                cropped_images.append({
                    "id": crop_id,
                    "type": item_type,
                    "confidence": item["confidence"],
                    "image_url": f"/crop/{os.path.basename(crop_dir)}/{crop_filename}"
                })
        
        # Run product identification on the cropped images
        prefilter_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"pre_filter_{unique_filename.split('.')[0]}")
        os.makedirs(prefilter_folder, exist_ok=True)
        
        try:
            # Run the product classification
            from modeloid import identify_products
            
            classification_result = identify_products(
                crop_dir, 
                prefilter_folder, 
                MODELS_DIR,
                "DIOS.pt"  # Make sure this is the correct model name
            )
            
            # Add classification results to the API response
            if "error" not in classification_result:
                # Create a mapping from crop ID to classification
                classification_map = {}
                for cls_item in classification_result.get("classifications", []):
                    classification_map[cls_item["id"]] = {
                        "predicted_class": cls_item.get("predicted_class", "unknown"),
                        "confidence": cls_item.get("confidence", 0.0)
                    }
                
                # Add classification info to each cropped image
                for crop in cropped_images:
                    crop_id = crop["id"]
                    if crop_id in classification_map and crop["type"] == "product":
                        crop["predicted_class"] = classification_map[crop_id]["predicted_class"]
                        crop["classification_confidence"] = classification_map[crop_id]["confidence"]
                        
                        # Create renamed copy of the image with class name in pre_filter folder
                        original_crop_path = os.path.join(crop_dir, f"{crop_id}_{unique_filename}")
                        if os.path.exists(original_crop_path):
                            class_name = classification_map[crop_id]["predicted_class"]
                            class_filename = f"{crop_id}_{class_name}.png"
                            class_path = os.path.join(prefilter_folder, class_filename)
                            
                            # Copy and rename the image
                            img = cv2.imread(original_crop_path)
                            if img is not None:
                                cv2.imwrite(class_path, img)
                
                # Add composite image path to result correctly
                if "composite_image_path" in classification_result:
                    composite_filename = os.path.basename(classification_result["composite_image_path"])
                    result["classification_composite"] = f"/classification/{composite_filename}"
                
                result["classification_summary"] = {
                    "num_products": classification_result.get("num_products", 0),
                    "num_empty_spaces": classification_result.get("num_empty_spaces", 0),
                }
            else:
                # If there was an error in classification
                result["classification_error"] = classification_result["error"]
                
        except Exception as e:
            # If classification fails, add error to result but continue
            result["classification_error"] = f"Product classification failed: {str(e)}"
        
        # Remove rendered_image from result dict to reduce response size
        rendered_image = result.pop("rendered_image")
        
        # Add the image URL and cropped images to the result
        result["image_url"] = f"/image/{output_filename}"
        result["cropped_images"] = cropped_images
        
        # Return the analysis results
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

@app.route('/image/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint to retrieve the analyzed image
    """
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Cannot retrieve image: {str(e)}"}), 404


@app.route('/analyze/base64', methods=['POST'])
def analyze_image_base64():
    """
    Alternative endpoint that accepts base64-encoded images and returns base64-encoded results
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image data found in request"}), 400
        
        # Decode the base64 image
        try:
            image_data = base64.b64decode(data['image'])
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({"error": "Could not decode image data"}), 400
        except Exception as e:
            return jsonify({"error": f"Error decoding image: {str(e)}"}), 400
        
        # Analyze the shelf products
        result = analyze_shelf_products(image, MODELS_DIR)
        
        # Check if there was an error in the analysis
        if "error" in result:
            return jsonify({"error": result["error"]}), 500
        
        # Convert the rendered image to base64
        _, buffer = cv2.imencode('.jpg', result["rendered_image"])
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        # Remove rendered_image from result dict
        result.pop("rendered_image")
        
        # Add base64 encoded image to result
        result["image_base64"] = base64_image
        
        # Return the analysis results
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def index():
    """Serve a simple HTML page for testing the API"""
    html ="""<!DOCTYPE html>
<html>
<head>
    <title>Shelf Analysis API</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .container { display: flex; margin-top: 20px; }
        .upload-section { flex: 1; }
        .results-section { flex: 2; margin-left: 20px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        button { background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
        button:hover { background-color: #45a049; }
        #preview { max-width: 100%; margin-top: 10px; }
        #analyzedImage { max-width: 100%; margin-top: 10px; }
        #resultsJson { white-space: pre-wrap; background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
        .crop-item { border: 1px solid #ddd; padding: 5px; margin: 5px; text-align: center; }
        .crop-label { margin-top: 5px; font-size: 0.9em; }
    </style>
</head>
<body>
    <h1>Shelf Analysis API</h1>
    <p>Upload an image to analyze shelf products and empty spaces.</p>
    
    <div class="container">
        <div class="upload-section">
            <div class="form-group">
                <label for="imageUpload">Select Image:</label>
                <input type="file" id="imageUpload" accept=".jpg,.jpeg,.png">
            </div>
            <div class="form-group">
                <button id="uploadBtn">Analyze Image</button>
            </div>
            <div>
                <h3>Preview:</h3>
                <img id="preview" src="" alt="Preview will appear here" style="display: none;">
            </div>
        </div>
        
        <div class="results-section">
            <h3>Analysis Results:</h3>
            <div id="loading" style="display: none;">Processing... Please wait.</div>
            <img id="analyzedImage" src="" alt="Analyzed image will appear here" style="display: none;">
            <h4>Detection Data:</h4>
            <pre id="resultsJson"></pre>
        </div>
    </div>
    
    <script>
        document.getElementById('imageUpload').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        document.getElementById('uploadBtn').addEventListener('click', function() {
            const fileInput = document.getElementById('imageUpload');
            if (!fileInput.files.length) {
                alert('Please select an image to upload.');
                return;
            }
            
            const file = fileInput.files[0];
            const formData = new FormData();
            formData.append('file', file);
            
            // Show loading message


            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzedImage').style.display = 'none';
            document.getElementById('resultsJson').textContent = '';
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('loading').style.display = 'none';
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display the analyzed image
                const analyzedImage = document.getElementById('analyzedImage');
                analyzedImage.src = data.image_url;
                analyzedImage.style.display = 'block';
                
                // Display the JSON results
                //document.getElementById('resultsJson').textContent = JSON.stringify(data, null, 2);
                
                // Display the cropped images
                displayCroppedImages(data);
                
                // Display classification results
                displayClassificationResults(data);
            })
            .catch(error => {
                document.getElementById('loading').style.display = 'none';
                alert('Error: ' + error);
            });
        });

        function displayCroppedImages(data) {
            // Create a container for cropped images if it doesn't exist
            let croppedContainer = document.getElementById('croppedImagesContainer');
            if (!croppedContainer) {
                croppedContainer = document.createElement('div');
                croppedContainer.id = 'croppedImagesContainer';
                croppedContainer.style.marginTop = '20px';
                document.querySelector('.results-section').appendChild(croppedContainer);
            } else {
                croppedContainer.innerHTML = ''; // Clear previous images
            }
            
            // Add a title
            const title = document.createElement('h4');
            title.textContent = 'Cropped Images:';
            croppedContainer.appendChild(title);
            
            // Create a flex container for the images
            const imagesGrid = document.createElement('div');
            imagesGrid.style.display = 'flex';
            imagesGrid.style.flexWrap = 'wrap';
            imagesGrid.style.gap = '10px';
            croppedContainer.appendChild(imagesGrid);
            
            // Add each cropped image
            data.cropped_images.forEach(crop => {
                const imgContainer = document.createElement('div');
                imgContainer.className = 'crop-item';
                imgContainer.dataset.id = crop.id; // Add data attribute for ID
                imgContainer.style.border = crop.type === 'product' ? '2px solid green' : '2px solid red';
                imgContainer.style.padding = '5px';
                imgContainer.style.textAlign = 'center';
                
                const img = document.createElement('img');
                img.src = crop.image_url;
                img.alt = crop.id;
                img.style.maxWidth = '150px';
                img.style.maxHeight = '150px';
                imgContainer.appendChild(img);
                
                const label = document.createElement('div');
                label.className = 'crop-label'; // Add class for later reference
                label.textContent = `${crop.id} (${crop.confidence.toFixed(2)})`;
                imgContainer.appendChild(label);
                
                imagesGrid.appendChild(imgContainer);
            });
        }
        
function displayClassificationResults(data) {
    if (data.classification_composite) {
        // Create a container for classification results
        let classificationContainer = document.getElementById('classificationContainer');
        if (!classificationContainer) {
            classificationContainer = document.createElement('div');
            classificationContainer.id = 'classificationContainer';
            classificationContainer.style.marginTop = '30px';
            document.querySelector('.results-section').appendChild(classificationContainer);
        } else {
            classificationContainer.innerHTML = ''; // Clear previous results
        }
        
        // Add a title
        const title = document.createElement('h3');
        title.textContent = 'Product Classification Results:';
        classificationContainer.appendChild(title);
        
        // Add classification summary
        const summary = document.createElement('p');
        summary.textContent = `${data.classification_summary.num_products} products classified, ${data.classification_summary.num_empty_spaces} empty spaces detected`;
        classificationContainer.appendChild(summary);
        
        // Add the composite image
        const img = document.createElement('img');
        img.src = data.classification_composite;
        img.alt = 'Classification Composite';
        img.style.maxWidth = '100%';
        classificationContainer.appendChild(img);
    }
    
    // Display classification error if any
    if (data.classification_error) {
        const errorMsg = document.createElement('div');
        errorMsg.textContent = `Classification Error: ${data.classification_error}`;
        errorMsg.style.color = 'red';
        document.querySelector('.results-section').appendChild(errorMsg);
    }
    
    // Update crop labels with classification info
    for (const crop of data.cropped_images || []) {
        if (crop.predicted_class && crop.type === 'product') {
            // Find the crop container
            const cropElements = document.querySelectorAll(`[data-id="${crop.id}"]`);
            if (cropElements.length > 0) {
                const labelElement = cropElements[0].querySelector('.crop-label');
                if (labelElement) {
                    labelElement.textContent = `${crop.id}: ${crop.predicted_class} (${crop.classification_confidence.toFixed(2)})`;
                    labelElement.style.color = 'green';
                }
            }
        }
    }
    
    // Add button to go to the filter page
    if (data.cropped_images && data.cropped_images.length > 0) {
        // Create a button to go to the filter page
        const filterBtn = document.createElement('button');
        filterBtn.textContent = 'Start Classification Filtering';
        filterBtn.style.backgroundColor = '#2196F3';
        filterBtn.style.color = 'white';
        filterBtn.style.border = 'none';
        filterBtn.style.padding = '10px 15px';
        filterBtn.style.marginTop = '20px';
        filterBtn.style.cursor = 'pointer';
        
        // Extract session ID correctly from the image URL
        const imagePath = data.cropped_images[0].image_url;
        const pathParts = imagePath.split('/');
        const folderName = pathParts[2]; // crops_UUID_filename
        // Extract just the UUID part (the hexadecimal string)
        const uuidPattern = /[0-9a-f]{32}/;
        const match = folderName.match(uuidPattern);
        const sessionId = match ? match[0] : folderName.replace('crops_', '');
        
        console.log("Session ID for filtering:", sessionId);
        
        filterBtn.onclick = function() {
            window.location.href = `/filter/${sessionId}`;
        };
        
        document.querySelector('.results-section').appendChild(filterBtn);
    }
}
    </script>
</body>
</html>"""
    return html

@app.route('/crop/<folder>/<filename>', methods=['GET'])
def get_crop(folder, filename):
    """
    Endpoint to retrieve a cropped image
    """
    try:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        file_path = os.path.join(folder_path, filename)
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Cannot retrieve cropped image: {str(e)}"}), 404

@app.route('/classification/<filename>', methods=['GET'])
def get_classification_image(filename):
    """
    Endpoint to retrieve the classification composite image
    """
    try:
        # Check in uploads directory for the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            return send_file(file_path, mimetype='image/jpeg')
        
        # If not found directly, check in subfolders
        for root, dirs, files in os.walk(app.config['UPLOAD_FOLDER']):
            if filename in files:
                return send_file(os.path.join(root, filename), mimetype='image/jpeg')
        
        return jsonify({"error": "Classification image not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Cannot retrieve classification image: {str(e)}"}), 404

@app.route('/prefilter/<folder>/<filename>', methods=['GET'])
def get_prefilter_image(folder, filename):
    """
    Endpoint to retrieve a pre-filtered classified image
    """
    try:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        file_path = os.path.join(folder_path, filename)
        
        print(f"Requested prefilter image: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            # Try looking for the file with any extension
            for ext in ['.jpg', '.jpeg', '.png']:
                alt_filename = filename.rsplit('.', 1)[0] + ext
                alt_path = os.path.join(folder_path, alt_filename)
                if os.path.exists(alt_path):
                    print(f"Found alternative file: {alt_path}")
                    return send_file(alt_path, mimetype=f'image/{ext[1:]}')
            
            return jsonify({"error": f"File not found: {filename} in {folder}"}), 404
        
        # Determine mimetype based on extension
        if filename.endswith('.jpg') or filename.endswith('.jpeg'):
            mimetype = 'image/jpeg'
        elif filename.endswith('.png'):
            mimetype = 'image/png'
        else:
            mimetype = 'application/octet-stream'
            
        return send_file(file_path, mimetype=mimetype)
    except Exception as e:
        print(f"Error serving prefilter image: {str(e)}")
        return jsonify({"error": f"Cannot retrieve pre-filtered image: {str(e)}"}), 404

@app.route('/filter/<session_id>', methods=['GET'])
def filter_images(session_id):
    """
    Page to filter and confirm product classifications
    """
    try:
        # Add some debug logging
        print(f"Filtering for session ID: {session_id}")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        
        # List all folders in the upload directory
        all_folders = os.listdir(app.config['UPLOAD_FOLDER'])
        print(f"All folders: {all_folders}")
        
        # First, find the correct pre_filter folder that contains this session_id
        prefilter_folder = None
        for folder in all_folders:
            if folder.startswith('pre_filter_') and session_id in folder:
                prefilter_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                print(f"Found prefilter folder: {prefilter_folder}")
                break
        
        # If not found, look for crop folders to see if any session has this ID
        if not prefilter_folder:
            print(f"No prefilter folder found, looking for crop folder with session ID: {session_id}")
            for folder in all_folders:
                if folder.startswith('crops_') and session_id in folder:
                    crop_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                    folder_base = folder.replace('crops_', '')
                    prefilter_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"pre_filter_{folder_base}")
                    print(f"Found crop folder: {crop_folder}")
                    print(f"Using prefilter folder: {prefilter_folder}")
                    
                    # Create the pre_filter folder if it doesn't exist
                    if not os.path.exists(prefilter_folder):
                        print(f"Creating prefilter folder: {prefilter_folder}")
                        os.makedirs(prefilter_folder, exist_ok=True)
                    
                    # Check if product identification has been run, if not, run it now
                    if os.path.exists(prefilter_folder) and (not os.path.isdir(prefilter_folder) or len(os.listdir(prefilter_folder)) == 0):
                        print(f"Prefilter folder is empty, running identify_products")
                        try:
                            # Import here to avoid circular imports
                            from modeloid import identify_products
                            
                            # List all files in the crop folder
                            crop_files = os.listdir(crop_folder)
                            print(f"Crop folder contents ({len(crop_files)} files): {crop_files[:10]}...")
                            
                            # Debug the crop folder
                            product_crops_count = sum(1 for f in crop_files if f.startswith('P'))
                            print(f"Number of product crops (P*): {product_crops_count}")
                            
                            # Run identify_products
                            result = identify_products(
                                crop_folder, 
                                prefilter_folder, 
                                MODELS_DIR,
                                "DIOS.pt"
                            )
                            print(f"identify_products result: {result}")
                            
                            # If identify_products didn't create files, manually create some for testing
                            if os.path.exists(prefilter_folder) and len(os.listdir(prefilter_folder)) == 0:
                                print("No files created by identify_products, manually copying crops")
                                for crop_file in crop_files:
                                    if crop_file.startswith('P'):
                                        src_path = os.path.join(crop_folder, crop_file)
                                        # Extract the product ID (e.g., P11)
                                        product_id = crop_file.split('_')[0]
                                        # Create a dummy class name
                                        class_name = "unknown"
                                        dst_filename = f"{product_id}_{class_name}.png"
                                        dst_path = os.path.join(prefilter_folder, dst_filename)
                                        # Copy the file
                                        shutil.copy2(src_path, dst_path)
                                        print(f"Manually copied {src_path} to {dst_path}")
                            
                        except Exception as e:
                            print(f"Error running identify_products: {str(e)}")
                            traceback.print_exc()
                    break
        
        if not prefilter_folder:
            error_msg = f"Session not found or no images to filter. Session ID: {session_id}"
            print(error_msg)
            return error_msg, 404
        
        if not os.path.exists(prefilter_folder):
            error_msg = f"Prefilter folder does not exist: {prefilter_folder}"
            print(error_msg)
            return error_msg, 404
        
        # Check if we have any images in the folder
        image_files = []
        
        if os.path.exists(prefilter_folder):
            for f in os.listdir(prefilter_folder):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(f)
            print(f"Found {len(image_files)} images in {prefilter_folder}")
            if len(image_files) > 0:
                print(f"Sample image files: {image_files[:5]}")
            else:
                print(f"No image files found in {prefilter_folder}")
                print(f"All files in folder: {os.listdir(prefilter_folder)}")
        
        if not image_files:
            # If no images found in the prefilter folder, manually copy them from the crops folder
            print(f"No images found in prefilter folder, manually copying from crops")
            
            # Find the crops folder
            crops_folder = None
            for folder in all_folders:
                if folder.startswith('crops_') and session_id in folder:
                    crops_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                    break
            
            if crops_folder and os.path.exists(crops_folder):
                print(f"Found crops folder: {crops_folder}")
                crop_files = os.listdir(crops_folder)
                print(f"Crop folder contents ({len(crop_files)} files): {crop_files[:10]}...")
                
                # Manually copy product images to prefilter folder
                for crop_file in crop_files:
                    if crop_file.startswith('P'):
                        src_path = os.path.join(crops_folder, crop_file)
                        # Extract the product ID (e.g., P11)
                        product_id = crop_file.split('_')[0]
                        # Create a dummy class name
                        class_name = "unknown"
                        dst_filename = f"{product_id}_{class_name}.png"
                        dst_path = os.path.join(prefilter_folder, dst_filename)
                        # Copy the file
                        shutil.copy2(src_path, dst_path)
                        print(f"Manually copied {src_path} to {dst_path}")
                        # Add to image_files list
                        image_files.append(dst_filename)
            
            else:
                error_msg = f"No crop folder found for session {session_id}"
                print(error_msg)
                return error_msg, 404
        
        if not image_files:
            error_msg = f"No images found or could be created in filter folder: {prefilter_folder}"
            print(error_msg)
            return error_msg, 404
        
        # Create final images folder if it doesn't exist
        final_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"final_{os.path.basename(prefilter_folder).replace('pre_filter_', '')}")
        os.makedirs(final_folder, exist_ok=True)
        print(f"Created final folder: {final_folder}")
        
        # Classes dictionary for selection
        classes = {
            "1": "atun_dolores_lata_grande",
            "2": "atun_dolores_lata_pequena",
            "3": "atun_mazatun_agua",
            "4": "atun_dolores_ensalada_maxi",
            "5": "atun_mazatun_ensalada",
            "6": "elotitos_dolores_surfy",
            "7": "aluminio_festivo_76m",
            "8": "azucar_posada_1kg",
            "9": "arroz_posada_900g",
            "10": "lechera_doypack_209g",
            "11": "leche_carnation_evaporada",
            "12": "leche_nutralat_1_5l",
            "13": "leche_nutralat_1l",
            "14": "cafe_andatti_descafeinado",
            "15": "cafe_andatti_soluble",
            "16": "aceite_posada_800ml",
            "17": "aceite_nutrioli_850ml",
            "18": "aceite_nutrioli_400ml",
            "19": "mantequilla_sabrosano",
            "20": "frijol_lasierra_chorizo",
            "21": "frijol_pinto_posada",
            "22": "papas_ruffles_queso",
            "23": "papas_barcel_sal",
            "24": "chicharron_bitz_natural",
            "25": "chicharron_bitz_intenso",
            "26": "cheetos_torciditos_pequeno",
            "27": "cheetos_torciditos_grande",
            "28": "takis_blue"
        }
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Product Classification Filter</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .debug-info { background-color: #f8f9fa; padding: 10px; margin-bottom: 20px; border: 1px solid #ddd; }
                .image-container { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
                .image-item { border: 1px solid #ddd; padding: 10px; width: 200px; }
                .image-item img { width: 100%; height: auto; }
                .image-name { font-weight: bold; margin: 10px 0; }
                .action-buttons { display: flex; gap: 10px; margin-top: 10px; }
                button { padding: 5px 10px; cursor: pointer; }
                .confirm-btn { background-color: #4CAF50; color: white; border: none; }
                .change-btn { background-color: #f44336; color: white; border: none; }
                .class-select { width: 100%; margin-top: 10px; display: none; }
                .progress { margin-top: 20px; padding: 10px; background-color: #f5f5f5; }
                .finish-btn { background-color: #2196F3; color: white; border: none; padding: 10px 15px; margin-top: 20px; }
                .error-log { color: red; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Product Classification Filter</h1>
            

            
            <div class="progress">
                <span id="processed">0</span> of <span id="total">0</span> images processed
            </div>
            
            <div class="image-container" id="imageContainer">
                <!-- Images will be inserted here by JavaScript -->
            </div>
            
            <div class="error-log" id="errorLog"></div>
            
            <button id="finishBtn" class="finish-btn" style="display:none;">Finish Classification</button>
            
            <script>
                // Class options for select dropdown
                const classes = {
                    CLASSES_JSON
                };
                
                // Session ID
                const sessionId = "SESSION_ID";
                const prefilterFolder = "PREFILTER_FOLDER";
                
                // Image files
                const imageFiles = IMAGE_FILES_JSON;
                
                // Setup debug info

                // Filter out composite image if it exists
                const filteredImageFiles = imageFiles.filter(file => !file.startsWith('composite_'));
                
                // Track progress
                let processed = 0;
                const total = filteredImageFiles.length;
                document.getElementById('total').textContent = total;
                
                // Load the first batch of images
                const batchSize = 10;
                let currentBatch = 0;
                
                // Function to load a batch of images
                function loadImageBatch() {
                    try {
                        const container = document.getElementById('imageContainer');
                        container.innerHTML = ''; // Clear container
                        
                        const start = currentBatch * batchSize;
                        const end = Math.min(start + batchSize, filteredImageFiles.length);
                        
                        if (start >= filteredImageFiles.length) {
                            document.getElementById('errorLog').textContent = "No more images to load";
                            return;
                        }
                        
                        for (let i = start; i < end; i++) {
                            const filename = filteredImageFiles[i];
                            
                            // Skip composite images
                            if (filename.startsWith('composite_')) continue;
                            
                            try {
                                // Parse the current class from filename (format: P11_classname.png)
                                const filenameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
                                const parts = filenameWithoutExt.split('_');
                                const itemId = parts[0]; // P11
                                const currentClass = parts.slice(1).join('_'); // classname
                                
                                // Create image item container
                                const item = document.createElement('div');
                                item.className = 'image-item';
                                item.dataset.filename = filename;
                                
                                // Create image
                                const img = document.createElement('img');
                                img.src = `/prefilter/${prefilterFolder}/${filename}`;
                                img.alt = itemId;
                                img.onerror = function() {
                                    this.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100'%3E%3Crect width='100' height='100' fill='%23f8f9fa'/%3E%3Ctext x='50%' y='50%' font-size='14' text-anchor='middle' dominant-baseline='middle' fill='%23dc3545'%3EImage Error%3C/text%3E%3C/svg%3E";
                                    document.getElementById('errorLog').textContent += `Error loading image: ${filename} `;
                                };
                                item.appendChild(img);
                                
                                // Create image name/ID
                                const nameDiv = document.createElement('div');
                                nameDiv.className = 'image-name';
                                nameDiv.textContent = `${itemId}: ${currentClass}`;
                                item.appendChild(nameDiv);
                                
                                // Create action buttons
                                const btnDiv = document.createElement('div');
                                btnDiv.className = 'action-buttons';
                                
                                // Confirm button
                                const confirmBtn = document.createElement('button');
                                confirmBtn.className = 'confirm-btn';
                                confirmBtn.textContent = 'Confirm';
                                confirmBtn.onclick = function() {
                                    processImage(filename, currentClass);
                                    item.style.opacity = '0.5';
                                    item.querySelector('.confirm-btn').disabled = true;
                                    item.querySelector('.change-btn').disabled = true;
                                };
                                btnDiv.appendChild(confirmBtn);
                                
                                // Change class button
                                const changeBtn = document.createElement('button');
                                changeBtn.className = 'change-btn';
                                changeBtn.textContent = 'Change';
                                changeBtn.onclick = function() {
                                    const select = item.querySelector('.class-select');
                                    select.style.display = select.style.display === 'block' ? 'none' : 'block';
                                };
                                btnDiv.appendChild(changeBtn);
                                
                                item.appendChild(btnDiv);
                                
                                // Create class select dropdown (hidden by default)
                                const select = document.createElement('select');
                                select.className = 'class-select';
                                
                                // Add option for each class
                                for (const [id, className] of Object.entries(classes)) {
                                    const option = document.createElement('option');
                                    option.value = className;
                                    option.textContent = `${id}: ${className}`;
                                    if (className === currentClass) {
                                        option.selected = true;
                                    }
                                    select.appendChild(option);
                                }
                                
                                // Handle selection change
                                select.onchange = function() {
                                    nameDiv.textContent = `${itemId}: ${select.value}`;
                                };
                                
                                item.appendChild(select);
                                
                                // Add submit button for class change
                                const submitBtn = document.createElement('button');
                                submitBtn.textContent = 'Submit';
                                submitBtn.style.display = 'none';
                                submitBtn.onclick = function() {
                                    processImage(filename, select.value);
                                    item.style.opacity = '0.5';
                                    item.querySelector('.confirm-btn').disabled = true;
                                    item.querySelector('.change-btn').disabled = true;
                                    select.disabled = true;
                                    submitBtn.disabled = true;
                                };
                                select.parentNode.insertBefore(submitBtn, select.nextSibling);
                                
                                // Show submit button when select is visible
                                select.addEventListener('change', function() {
                                    submitBtn.style.display = 'block';
                                });
                                
                                container.appendChild(item);
                            } catch (error) {
                                document.getElementById('errorLog').textContent += `Error processing image ${filename}: ${error.message} `;
                            }
                        }
                        
                        // Show finish button if this is the last batch
                        if (end >= filteredImageFiles.length && processed >= total) {
                            document.getElementById('finishBtn').style.display = 'block';
                        }
                    } catch (error) {
                        document.getElementById('errorLog').textContent = `Batch loading error: ${error.message}`;
                    }
                }
                
                // Process an image (confirm or change class)
                function processImage(filename, className) {
                    fetch('/process_image', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: sessionId,
                            prefilter_folder: prefilterFolder,
                            filename: filename,
                            new_class: className
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            processed++;
                            document.getElementById('processed').textContent = processed;
                            
                            // Check if all images in the current batch are processed
                            const batchImages = document.querySelectorAll('.image-item');
                            const allProcessed = Array.from(batchImages).every(item => 
                                item.querySelector('.confirm-btn').disabled);
                            
                            if (allProcessed) {
                                currentBatch++;
                                if (currentBatch * batchSize < filteredImageFiles.length) {
                                    loadImageBatch();
                                } else {
                                    // Show finish button when all images are processed
                                    document.getElementById('finishBtn').style.display = 'block';
                                }
                            }
                        } else {
                            alert('Error: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('errorLog').textContent += `Process error: ${error.message} `;
                    });
                }
                
                // Initialize
                try {
                    if (filteredImageFiles.length > 0) {
                        loadImageBatch();
                    } else {
                        document.getElementById('errorLog').textContent = "No product images found (filtered out composite images)";
                    }
                } catch (error) {
                    document.getElementById('errorLog').textContent = `Initialization error: ${error.message}`;
                }
                
                // Finish button handler
                document.getElementById('finishBtn').addEventListener('click', function() {
                    window.location.href = '/results/' + sessionId;
                });
            </script>
        </body>
        </html>
        """
        
        # Replace placeholders with actual data
        html = html.replace("CLASSES_JSON", json.dumps(classes))
        html = html.replace("SESSION_ID", session_id)
        html = html.replace("PREFILTER_FOLDER", os.path.basename(prefilter_folder))
        html = html.replace("IMAGE_FILES_JSON", json.dumps(image_files))
        html = html.replace("IMAGE_COUNT", str(len(image_files)))
        
        print(f"Generated HTML with {len(image_files)} images")
        return html
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        error_msg = f"Error in filter_images: {str(e)}\n{traceback_str}"
        print(error_msg)
        return error_msg, 500


@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Process a single image classification (confirm or change)
    """
    try:
        data = request.get_json()
        
        if not data or 'session_id' not in data or 'filename' not in data or 'new_class' not in data:
            return jsonify({"success": False, "error": "Missing required data"}), 400
        
        session_id = data['session_id']
        filename = data['filename']
        new_class = data['new_class']
        
        # Use the prefilter_folder if provided
        if 'prefilter_folder' in data and data['prefilter_folder']:
            prefilter_folder = os.path.join(app.config['UPLOAD_FOLDER'], data['prefilter_folder'])
        else:
            # Find the correct prefilter folder
            prefilter_folder = None
            for folder in os.listdir(app.config['UPLOAD_FOLDER']):
                if folder.startswith('pre_filter_') and session_id in folder:
                    prefilter_folder = os.path.join(app.config['UPLOAD_FOLDER'], folder)
                    break
            
            if not prefilter_folder:
                return jsonify({"success": False, "error": f"Prefilter folder not found for session {session_id}"}), 404
        
        # Destination folder
        final_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"final_{os.path.basename(prefilter_folder).replace('pre_filter_', '')}")
        os.makedirs(final_folder, exist_ok=True)
        
        # Check if file exists
        source_path = os.path.join(prefilter_folder, filename)
        if not os.path.exists(source_path):
            return jsonify({"success": False, "error": f"File {filename} not found in {prefilter_folder}"}), 404
        
        # Get the ID part of the filename (e.g., P11)
        id_part = filename.split('_')[0]
        
        # Create new filename with the confirmed/changed class
        new_filename = f"{id_part}_{new_class}.png"
        destination_path = os.path.join(final_folder, new_filename)
        
        # Copy the image to the final folder with the new filename
        shutil.copy2(source_path, destination_path)
        
        return jsonify({"success": True})
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/results/<session_id>', methods=['GET'])
def show_results(session_id):
    """
    Show final classification results
    """
    try:
        # Get the final images folder
        final_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"final_{session_id}")
        
        if not os.path.exists(final_folder):
            return "No final images found for this session.", 404
        
        # Get all image files
        image_files = []
        for f in os.listdir(final_folder):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(f)
        
        # Sort by ID
        image_files.sort(key=lambda x: x.split('_')[0])
        
        # Generate HTML for results page
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Final Classification Results</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .image-grid { display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px; }
                .image-item { border: 1px solid #ddd; padding: 10px; width: 200px; }
                .image-item img { width: 100%; height: auto; }
                .image-name { font-weight: bold; margin: 10px 0; text-align: center; }
                .download-btn { background-color: #2196F3; color: white; border: none; padding: 10px 15px; 
                               margin-top: 20px; cursor: pointer; }
            </style>
        </head>
        <body>
            <h1>Final Classification Results</h1>
            <p>Total images: <span id="totalCount">0</span></p>
            <button id="downloadBtn" class="download-btn">Download Results as JSON</button>
            
            <div class="image-grid" id="imageGrid">
                <!-- Images will be inserted here by JavaScript -->
            </div>
            
            <script>
                // Image files
                const imageFiles = IMAGE_FILES_JSON;
                document.getElementById('totalCount').textContent = imageFiles.length;
                
                // Session ID
                const sessionId = "SESSION_ID";
                
                // Display images
                const grid = document.getElementById('imageGrid');
                
                imageFiles.forEach(filename => {
                    // Parse the ID and class from filename
                    const filenameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
                    const parts = filenameWithoutExt.split('_');
                    const itemId = parts[0]; // P11
                    const className = parts.slice(1).join('_'); // classname
                    
                    // Create image item
                    const item = document.createElement('div');
                    item.className = 'image-item';
                    
                    // Create image
                    const img = document.createElement('img');
                    img.src = `/final/final_${sessionId}/${filename}`;
                    img.alt = itemId;
                    item.appendChild(img);
                    
                    // Create label
                    const label = document.createElement('div');
                    label.className = 'image-name';
                    label.textContent = `${itemId}: ${className}`;
                    item.appendChild(label);
                    
                    grid.appendChild(item);
                });
                
                // Download results as JSON
                document.getElementById('downloadBtn').addEventListener('click', function() {
                    const results = imageFiles.map(filename => {
                        const filenameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
                        const parts = filenameWithoutExt.split('_');
                        const itemId = parts[0];
                        const className = parts.slice(1).join('_');
                        
                        return {
                            id: itemId,
                            class: className,
                            filename: filename
                        };
                    });
                    
                    const jsonStr = JSON.stringify(results, null, 2);
                    const blob = new Blob([jsonStr], { type: 'application/json' });
                    const url = URL.createObjectURL(blob);
                    
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `classification_results_${sessionId}.json`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                });
            </script>
        </body>
        </html>
        """
        
        # Replace placeholders
        html = html.replace("IMAGE_FILES_JSON", json.dumps(image_files))
        html = html.replace("SESSION_ID", session_id)
        
        return html
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/final/<folder>/<filename>', methods=['GET'])
def get_final_image(folder, filename):
    """
    Endpoint to retrieve a final classified image
    """
    try:
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder)
        file_path = os.path.join(folder_path, filename)
        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": f"Cannot retrieve final image: {str(e)}"}), 404

if __name__ == '__main__':
    # Check if models exist before starting the server
    yolov5_model = os.path.join(MODELS_DIR, 'jonathan.pt')
    yolov8_model = os.path.join(MODELS_DIR, 'empty_spaces.pt')
    
    if not os.path.isfile(yolov5_model) or not os.path.isfile(yolov8_model):
        print("Warning: Required models not found. Please ensure the models are available at:")
        print(f"- YOLOv5 model: {yolov5_model}")
        print(f"- YOLOv8 model: {yolov8_model}")
    
    # Start the server
    app.run(host='0.0.0.0', port=5000, debug=True)