import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
import gc
from typing import Dict, List, Tuple
import re
from dotenv import load_dotenv
from google import genai

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load environment variables
load_dotenv()

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing spaces
    text = text.strip()
    # Remove multiple dots
    text = re.sub(r'\.+', '.', text)
    # Remove dots at the start of lines
    text = re.sub(r'^\.+', '', text)
    return text

# Initialize Google Generative AI client
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Configure the API
client = genai.Client(api_key=GOOGLE_API_KEY)

def summarize_text(text: str) -> str:
    """
    Summarize text using Google's Gemini model.
    Args:
        text: Text to summarize
    Returns:
        Summarized text
    """
    if not text.strip():
        return ""
        
    try:
        # Create the summarization prompt
        prompt = f"""Please provide a concise summary of the following text. 
Keep the same technical terminology and maintain the key information.

TEXT:
{text}

SUMMARY:"""
        
        # Generate the summary using the flash model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Return the generated text
        return response.text if response.text else "Error: Unable to generate summary"
    except Exception as e:
        print(f"Summarization error: {str(e)}")  # Added logging
        return f"Summarization failed: {str(e)}"

# Initialize models
# Initialize PaddleOCR with optimized parameters for handwriting detection
try:
    ocr = PaddleOCR(
        use_angle_cls=False,           # Disable angle classification for simplicity
        lang='en',
        use_gpu=True,
        det_algorithm='DB',            # DB algorithm for detection
        det=True,                      # Enable detection
        rec=False,                     # Disable recognition since we're using TrOCR
        
        # Optimized detection parameters for handwriting
        det_db_thresh=0.25,            # Lower threshold to detect faint strokes
        det_db_box_thresh=0.35,        # Lower box threshold for messy/faint text
        det_db_unclip_ratio=1.5,       # Larger boxes for loose handwriting
        det_db_use_dilation=True,      # Help connect broken strokes
        det_db_box_type='poly',        # Polygon boxes for slanted writing
        det_db_score_mode='slow',      # More accurate scoring
        
        # Additional parameters
        det_max_candidates=2000,       # Allow more text regions
        det_limit_side_len=1280,       # Higher resolution limit
        det_limit_type='max',          # Maintain aspect ratio
        
        # Use PP-OCRv3 detection model which works well with handwriting
        det_model_dir='ch_PP-OCRv4_det_infer'
    )
except Exception as e:
    raise Exception(f"Error initializing PaddleOCR: {str(e)}")

try:
    # Initialize TrOCR with handwritten-specific model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    model = model.to(device)
except Exception as e:
    raise Exception(f"Error loading TrOCR model: {str(e)}")

def sort_boxes_by_position(texts):
    """
    Sort text boxes in reading order (top to bottom, then left to right).
    Uses a line grouping approach where boxes close in y-coordinate are considered on the same line.
    """
    if not texts:
        return texts
        
    # First, group boxes into lines based on vertical position
    line_threshold = 20  # pixels - boxes within this y-distance are considered on same line
    lines = {}
    
    for text in texts:
        bbox = text["bbox"]
        # Get center y-coordinate of the box
        y_center = sum(point[1] for point in bbox) / len(bbox)
        
        # Find a line that this box belongs to
        assigned = False
        for line_idx, line_y in lines.keys():
            if abs(y_center - line_y) < line_threshold:
                lines[(line_idx, line_y)].append(text)
                assigned = True
                break
        
        # If no suitable line found, create new line
        if not assigned:
            new_line_idx = len(lines)
            lines[(new_line_idx, y_center)] = [text]
    
    # Sort lines by y-coordinate
    sorted_lines = sorted(lines.items(), key=lambda x: x[0][1])
    
    # Within each line, sort boxes left to right
    result = []
    for _, line_boxes in sorted_lines:
        # Sort boxes in this line by x_percent (left to right)
        line_boxes.sort(key=lambda x: x["x_percent"])
        result.extend(line_boxes)
    
    # Update region numbers to match new order
    for i, text in enumerate(result):
        text["region"] = i + 1
    
    return result

def fix_text_errors(text: str) -> str:
    """
    Fix errors in the text while maintaining the original structure using Gemini AI.
    Args:
        text: Text to fix
    Returns:
        Corrected text while maintaining structure
    """
    if not text.strip():
        return ""
        
    try:
        prompt = f"""Please fix any spelling, grammar, or formatting errors in the following text.
Keep the same structure, terminology, and meaning. Only fix obvious errors.
Do not summarize or change the content. Return the corrected version with the same format.

TEXT TO FIX:
{text}

CORRECTED TEXT:"""

        # Generate the corrected text using the flash model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text if response.text else text  # Return original if AI fails
    except Exception as e:
        print(f"Text correction failed: {str(e)}")
        return text  # Return original text if there's an error

def calculate_horizontal_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for horizontal overlap between two boxes.
    Only considers horizontal overlap, vertical overlap is ignored.
    Args:
        box1: First box coordinates as numpy array
        box2: Second box coordinates as numpy array
    Returns:
        Horizontal IoU value between 0 and 1
    """
    # Get x-coordinates
    x1_min, x1_max = np.min(box1[:, 0]), np.max(box1[:, 0])
    x2_min, x2_max = np.min(box2[:, 0]), np.max(box2[:, 0])
    
    # Calculate intersection
    intersection = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    
    # Calculate union
    union = (x1_max - x1_min) + (x2_max - x2_min) - intersection
    
    # Return IoU
    return intersection / union if union > 0 else 0

def has_horizontal_overlap(box1: np.ndarray, box2: np.ndarray) -> bool:
    """
    Check if two boxes overlap horizontally.
    Args:
        box1: First box coordinates as numpy array
        box2: Second box coordinates as numpy array
    Returns:
        True if boxes overlap horizontally
    """
    x1_min, x1_max = np.min(box1[:, 0]), np.max(box1[:, 0])
    x2_min, x2_max = np.min(box2[:, 0]), np.max(box2[:, 0])
    
    # Check if one box's x-range overlaps with the other
    return (x1_min <= x2_max and x2_min <= x1_max)

def should_merge_boxes(box1: np.ndarray, box2: np.ndarray, 
                      max_horizontal_gap: int = 50) -> bool:
    """
    Determine if two boxes should be merged based on horizontal proximity.
    Args:
        box1: First box coordinates
        box2: Second box coordinates
        max_horizontal_gap: Maximum pixel gap between boxes
    Returns:
        True if boxes should be merged
    """
    # Get y-coordinates to check if boxes are on same line
    y1_center = np.mean(box1[:, 1])
    y2_center = np.mean(box2[:, 1])
    
    # Boxes must be within vertical threshold to be considered on same line
    vertical_threshold = 20  # pixels
    if abs(y1_center - y2_center) > vertical_threshold:
        return False
    
    # Check if boxes overlap horizontally
    if has_horizontal_overlap(box1, box2):
        return True
    
    # If boxes don't overlap but are close horizontally
    x1_max = np.max(box1[:, 0])
    x2_min = np.min(box2[:, 0])
    gap = x2_min - x1_max
    
    return 0 <= gap <= max_horizontal_gap

def merge_box_pair(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """
    Merge two boxes into one larger box.
    Args:
        box1: First box coordinates
        box2: Second box coordinates
    Returns:
        Merged box coordinates
    """
    # Get the extremes of both boxes
    x_min = min(np.min(box1[:, 0]), np.min(box2[:, 0]))
    x_max = max(np.max(box1[:, 0]), np.max(box2[:, 0]))
    y_min = min(np.min(box1[:, 1]), np.min(box2[:, 1]))
    y_max = max(np.max(box1[:, 1]), np.max(box2[:, 1]))
    
    # Create new box coordinates
    return np.array([
        [x_min, y_min],  # Top-left
        [x_max, y_min],  # Top-right
        [x_max, y_max],  # Bottom-right
        [x_min, y_max]   # Bottom-left
    ], dtype=np.int32)

def merge_overlapping_boxes(boxes: list) -> list:
    """
    Merge horizontally overlapping or nearby boxes while maintaining line separation.
    Ensures no horizontal overlap between final boxes.
    Args:
        boxes: List of box coordinates
    Returns:
        List of merged box coordinates with no horizontal overlap
    """
    if not boxes:
        return boxes
    
    # Convert boxes to numpy arrays if they aren't already
    boxes = [np.array(box) for box in boxes]
    
    # Group boxes by lines based on y-coordinate
    line_groups = {}
    for i, box in enumerate(boxes):
        y_center = np.mean(box[:, 1])
        
        # Find the line this box belongs to
        assigned = False
        for line_y in line_groups:
            if abs(y_center - line_y) < 20:  # 20 pixels threshold for same line
                line_groups[line_y].append((i, box))
                assigned = True
                break
        
        if not assigned:
            line_groups[y_center] = [(i, box)]
    
    # Process each line separately
    final_boxes = []
    
    for line_y in sorted(line_groups.keys()):
        line_boxes = line_groups[line_y]
        
        # Sort boxes in this line by x-coordinate
        line_boxes.sort(key=lambda x: np.min(x[1][:, 0]))
        
        # Merge overlapping boxes in this line
        current_boxes = []
        current_box = line_boxes[0][1]
        
        for i in range(1, len(line_boxes)):
            next_box = line_boxes[i][1]
            
            if has_horizontal_overlap(current_box, next_box) or \
               should_merge_boxes(current_box, next_box):
                # Merge the boxes
                current_box = merge_box_pair(current_box, next_box)
            else:
                # No overlap or proximity, save current box and start new one
                current_boxes.append(current_box)
                current_box = next_box
        
        # Add the last box
        current_boxes.append(current_box)
        final_boxes.extend(current_boxes)
    
    return final_boxes

def merge_paddle_boxes_to_lines(
    boxes: List[List[List[int]]],
    y_overlap_threshold: float = 0.4, # Relative vertical overlap allowed (tune 0.3-0.5)
    x_proximity_threshold: int = 75,  # Max horizontal pixel gap allowed within a line (tune 50-100)
    max_height_ratio_filter: float = 2.5, # Filter boxes > X * median height (tune 2.0-3.5)
    min_box_height: int = 5          # Minimum height in pixels to consider a box
) -> List[Tuple[int, int, int, int]]:
    """
    Merges PaddleOCR detection boxes (polygons) into robust line boxes (rectangles),
    filtering out excessively tall boxes and grouping based on vertical and horizontal proximity.
    (This is the recommended version from the previous answer)
    """
    if not boxes:
        return []

    # 1. Initial Conversion & Filtering
    rects_data = [] # Store [x_min, y_min, x_max, y_max, y_center, height]
    heights = []
    for box in boxes:
        try:
            points = np.array(box, dtype=np.int32)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            height = y_max - y_min
            width = x_max - x_min

            if height >= min_box_height and width > 0: # Basic validity check
                y_center = y_min + height / 2.0
                rects_data.append([x_min, y_min, x_max, y_max, y_center, height])
                heights.append(height)
        except Exception as e:
            # print(f"Warning: Could not process box {box}: {e}") # Optional logging
            continue

    if not rects_data:
        return []

    # --- Filter excessively tall boxes ---
    filtered_rects = []
    if max_height_ratio_filter is not None and heights:
        median_height = np.median(heights)
        max_allowed_height = median_height * max_height_ratio_filter
        for rect in rects_data:
            if rect[5] <= max_allowed_height:
                filtered_rects.append(rect)
    else:
        filtered_rects = rects_data # No height filtering

    if not filtered_rects:
        return []

    # 2. Sorting
    filtered_rects.sort(key=lambda r: (r[4], r[0])) # Sort by y_center, then x_min

    # 3. Line Grouping (Iterative Merging)
    merged_lines_boxes = []
    current_line_rects = [] # Stores the raw rects for the current line

    for rect in filtered_rects:
        x_min, y_min, x_max, y_max, y_center, height = rect

        if not current_line_rects:
            current_line_rects.append(rect)
        else:
            avg_y_of_current_line = sum(r[4] for r in current_line_rects) / len(current_line_rects)
            max_height_of_current_line = max(r[5] for r in current_line_rects)
            rightmost_x_of_current_line = max(r[2] for r in current_line_rects)

            # Check vertical alignment (baseline check using y_center and height)
            vertically_aligned = abs(y_center - avg_y_of_current_line) < max(height, max_height_of_current_line) * y_overlap_threshold

            # Check horizontal proximity (gap check)
            horizontal_gap = x_min - rightmost_x_of_current_line
            horizontally_proximate = horizontal_gap < x_proximity_threshold

            if vertically_aligned and horizontally_proximate:
                current_line_rects.append(rect)
            else:
                # 4. Finalize the previous line
                if current_line_rects:
                    line_x_min = min(r[0] for r in current_line_rects)
                    line_y_min = min(r[1] for r in current_line_rects)
                    line_x_max = max(r[2] for r in current_line_rects)
                    line_y_max = max(r[3] for r in current_line_rects)
                    if line_x_max > line_x_min and line_y_max > line_y_min:
                        merged_lines_boxes.append((line_x_min, line_y_min, line_x_max, line_y_max))
                # Start a new line
                current_line_rects = [rect]

    # Add the last line being built
    if current_line_rects:
        line_x_min = min(r[0] for r in current_line_rects)
        line_y_min = min(r[1] for r in current_line_rects)
        line_x_max = max(r[2] for r in current_line_rects)
        line_y_max = max(r[3] for r in current_line_rects)
        if line_x_max > line_x_min and line_y_max > line_y_min:
             merged_lines_boxes.append((line_x_min, line_y_min, line_x_max, line_y_max))

    # 5. Final sort (optional, should be mostly sorted)
    merged_lines_boxes.sort(key=lambda line: line[1]) # Sort by y_min

    return merged_lines_boxes

def extract_text(image_path: str, visualization_dir: str = None, use_improvement: bool = False, debug_visualization: bool = False) -> Dict:
    """
    Extract text from an image file using PaddleOCR for detection and TrOCR for recognition.
    Args:
        image_path: Path to the image file
        visualization_dir: Directory to save visualization output
        use_improvement: Whether to summarize the extracted text using AI
        debug_visualization: Whether to create visualization for debugging
    Returns:
        Dictionary containing extracted texts and metadata
    """
    try:
        # Normalize path and check if file exists
        image_path = os.path.abspath(os.path.normpath(image_path))
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
            
        try:
            # Try to read the file into a buffer
            with open(image_path, 'rb') as f:
                file_bytes = f.read()
                
            if not file_bytes:
                return {"error": f"Image file is empty: {image_path}"}
                
            # Convert to numpy array
            nparr = np.frombuffer(file_bytes, np.uint8)
            if nparr.size == 0:
                return {"error": f"Failed to convert image to numpy array: {image_path}"}
                
            # Decode the image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                # Try reading with PIL as fallback
                try:
                    pil_image = Image.open(image_path)
                    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                except Exception as pil_error:
                    return {"error": f"Could not read image with either OpenCV or PIL: {image_path}. Error: {str(pil_error)}"}
        except Exception as read_error:
            return {"error": f"Failed to read image file: {image_path}. Error: {str(read_error)}"}
        
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        
        # Create visualization directory only if debug_visualization is True
        if debug_visualization and visualization_dir:
            os.makedirs(visualization_dir, exist_ok=True)
        
        # Run PaddleOCR detection
        try:
            # Get detection results
            result = ocr.ocr(image, det=True, rec=False, cls=False)
            
            if not result or not result[0]:
                return {
                    "success": False,
                    "texts": [],
                    "message": "No text regions detected in the image."
                }
            
            # Create visualization only if debug_visualization is True
            visualization = image.copy() if debug_visualization else None
            
            # Merge boxes into lines
            boxes = result[0]
            merged_lines = merge_paddle_boxes_to_lines(boxes)
            
            # Process each detected line with TrOCR
            extracted_texts = []
            
            for i, line_box in enumerate(merged_lines):
                try:
                    x1, y1, x2, y2 = line_box
                    
                    # Add minimal padding (8 pixels for handwriting)
                    padding = 8
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image_width, x2 + padding)
                    y2 = min(image_height, y2 + padding)
                    
                    # Draw bounding box on visualization only if debug_visualization is True
                    if debug_visualization:
                        points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                        cv2.polylines(visualization, [points], True, (0, 255, 0), 2)
                        cv2.putText(visualization, str(i+1), (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Crop region
                    crop = image[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue
                    
                    # Convert to RGB for TrOCR
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_pil = Image.fromarray(crop_rgb)
                    
                    # Process with TrOCR
                    pixel_values = processor(crop_pil, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(device)
                    
                    with torch.no_grad():
                        generated_ids = model.generate(
                            pixel_values,
                            max_length=128,
                            num_beams=5,
                            early_stopping=True
                        )
                    
                    trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    trocr_text = clean_text(trocr_text)
                    
                    if trocr_text.strip():
                        # Calculate center point for y-coordinate sorting
                        y_center = (y1 + y2) / 2
                        # Convert all coordinates to standard Python integers
                        bbox = [[int(x1), int(y1)], [int(x2), int(y1)], 
                               [int(x2), int(y2)], [int(x1), int(y2)]]
                        extracted_texts.append({
                            "text": trocr_text,
                            "region": i+1,
                            "confidence": 1.0,
                            "bbox": bbox,
                            "x_percent": float((x1 + x2) / (2 * image_width)),
                            "y_center": float(y_center)
                        })
                except Exception as e:
                    print(f"Error processing region {i+1}: {str(e)}")
                    continue
            
            # Save visualization only if debug_visualization is True
            if debug_visualization and visualization_dir and visualization is not None:
                vis_path = os.path.join(visualization_dir, 'visualization.png')
                cv2.imwrite(vis_path, visualization)
            
            # Combine all texts
            combined_text = " ".join(text["text"] for text in extracted_texts)
            
            # Fix errors in the combined text
            corrected_text = fix_text_errors(combined_text)
            
            # Apply AI improvement/summary if requested
            if use_improvement and corrected_text.strip():
                improved_text = summarize_text(corrected_text)
            else:
                improved_text = ""
            
            return {
                "success": True,
                "texts": extracted_texts,
                "combined_text": combined_text,
                "corrected_text": corrected_text,
                "improved_text": improved_text,
                "visualization": "visualization.png" if debug_visualization and visualization_dir else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"OCR processing failed: {str(e)}"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": f"Image processing failed: {str(e)}"
        }
    finally:
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache() 