import cv2
import numpy as np
import matplotlib.pyplot as plt



def detect_puzzle_pieces(img):
    """
    New function that mimics the API of detect_pieces.py
    
    Args:
        img: Input image containing puzzle pieces (numpy array)
    
    Returns:
        pieces: List of individual piece images
        contours: List of contours for each piece  
        bboxes: List of bounding boxes (x, y, w, h) for each piece
    """
    if img is None:
        raise ValueError("Input image is None")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Noise reduction (Gaussian blur)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Edge detection with Canny
    edges = cv2.Canny(blurred, 0.1, 90)

    # 5. Morphological closing to improve edge continuity
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 6. Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 7. Filter out contours that are too small
    min_area = 560  
    filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # Sort contours by position (top-left to bottom-right)
    bboxes = []
    for c in filtered_contours:
        x, y, w, h = cv2.boundingRect(c)
        bboxes.append((x, y, w, h))
    
    # Sort by position (y, x)
    if bboxes:
        indices = np.lexsort(([b[0] for b in bboxes], [b[1] for b in bboxes]))
        filtered_contours = [filtered_contours[i] for i in indices]
        bboxes = [bboxes[i] for i in indices]
    
    # Extract each piece as a separate image
    pieces = []
    for c, (x, y, w, h) in zip(filtered_contours, bboxes):
        # Create a mask for this piece
        piece_mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(piece_mask, [c], -1, 255, -1)
        
        # Apply the mask to get just this piece
        piece = np.zeros_like(img)
        piece = cv2.bitwise_and(img, img, mask=piece_mask)
        
        # Crop to bounding box with a small margin
        margin = 5
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(width, x + w + margin)
        y_end = min(height, y + h + margin)
        cropped_piece = piece[y_start:y_end, x_start:x_end]
        
        pieces.append(cropped_piece)
    
    return pieces, filtered_contours, bboxes