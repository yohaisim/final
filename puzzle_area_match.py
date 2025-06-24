import cv2
import numpy as np
import os
import json
import base64
from PIL import Image
import io
from collections import Counter
from skimage.metrics import structural_similarity as ssim


def save_panoptic_data(segments_data, panoptic_mask, session_id, results_folder, segmented_image=None):
    """Save panoptic segmentation data"""
    try:
        panoptic_dir = os.path.join(results_folder, f"panoptic_data_{session_id}")
        os.makedirs(panoptic_dir, exist_ok=True)
        
        with open(os.path.join(panoptic_dir, "segments_info.json"), 'w') as f:
            json.dump(segments_data, f)
        
        np.save(os.path.join(panoptic_dir, "panoptic_mask.npy"), panoptic_mask)
        
        # Save the segmented image array directly to avoid color conversion issues
        if segmented_image is not None:
            np.save(os.path.join(panoptic_dir, "segmented_image.npy"), segmented_image)
        
        print(f"Saved panoptic data with {len(segments_data)} segments")
        return True
    except Exception as e:
        print(f"Error saving panoptic data: {e}")
        return False

def load_panoptic_data(session_id, results_folder):
    """Load panoptic segmentation data"""
    try:
        panoptic_dir = os.path.join(results_folder, f"panoptic_data_{session_id}")
        
        with open(os.path.join(panoptic_dir, "segments_info.json"), 'r') as f:
            segments_data = json.load(f)
        
        panoptic_mask = np.load(os.path.join(panoptic_dir, "panoptic_mask.npy"))
        
        # Try to load the saved segmented image
        segmented_image = None
        segmented_image_path = os.path.join(panoptic_dir, "segmented_image.npy")
        if os.path.exists(segmented_image_path):
            segmented_image = np.load(segmented_image_path)
        
        return {
            'segments_data': segments_data,
            'panoptic_mask': panoptic_mask,
            'segmented_image': segmented_image
        }
    except Exception as e:
        print(f"Error loading panoptic data: {e}")
        return None

def region_crop(original_rgb, mask, margin=4):
    """Return a tight RGB crop of the region described by mask."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    
    # Add margin but keep within bounds
    x0, y0 = max(0, x0-margin), max(0, y0-margin)
    x1, y1 = min(original_rgb.shape[1]-1, x1+margin), min(original_rgb.shape[0]-1, y1+margin)
    
    return original_rgb[y0:y1+1, x0:x1+1]

def texture_similarity(piece_bgr, seg_id, panoptic_mask, full_rgb):
    """SSIM similarity in luminance between piece and candidate region."""
    try:
        region_mask = panoptic_mask == seg_id
        region_rgb = region_crop(full_rgb, region_mask)
        
        if region_rgb is None or region_rgb.size == 0:
            return 0.0
        
        # Resize both to same size for comparison
        h, w = 120, 120
        
        # Convert piece to grayscale
        piece_resized = cv2.resize(piece_bgr, (w, h))
        piece_gray = cv2.cvtColor(piece_resized, cv2.COLOR_BGR2GRAY)
        
        # Convert region to grayscale
        region_resized = cv2.resize(region_rgb, (w, h))
        region_gray = cv2.cvtColor(region_resized, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        similarity = ssim(piece_gray, region_gray)
        return max(0.0, similarity)  # Ensure non-negative
        
    except Exception as e:
        print(f"Error in texture similarity calculation: {e}")
        return 0.0

def extract_individual_regions(original_image, panoptic_mask, segments_data):
    """Extract and save individual regions - not needed for this approach but keeping for compatibility"""
    individual_regions = []
    
    for segment_info in segments_data:
        seg_id = segment_info["seg_id"]
        class_name = segment_info["class_name"]
        
        segment_mask = (panoptic_mask == seg_id).astype(np.uint8) * 255
        
        if np.sum(segment_mask) < 1000:
            continue
        
        region_image = original_image.copy()
        mask_3ch = cv2.cvtColor(segment_mask, cv2.COLOR_GRAY2BGR)
        mask_3ch = mask_3ch.astype(np.float32) / 255.0
        region_image = (region_image.astype(np.float32) * mask_3ch).astype(np.uint8)
        
        individual_regions.append({
            'seg_id': seg_id,
            'class_name': class_name,
            'image': region_image
        })
    
    return individual_regions

def save_individual_regions(individual_regions, session_id, results_folder):
    """Save individual regions - keeping for compatibility"""
    return []

def create_side_by_side_image(piece_image, segmented_image_path):
    """Create combined image showing piece next to segmented puzzle"""
    # Convert piece from OpenCV to PIL
    piece_rgb = cv2.cvtColor(piece_image, cv2.COLOR_BGR2RGB)
    piece_pil = Image.fromarray(piece_rgb)
    
    # Load segmented image
    segmented_pil = Image.open(segmented_image_path).convert("RGB")
    segmented_pil = segmented_pil.resize((400, 300))
    
    # Create combined image
    combined_width = piece_pil.width + segmented_pil.width + 20
    combined_height = max(piece_pil.height, segmented_pil.height)
    combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    
    # Paste images side by side
    combined.paste(piece_pil, (0, 0))
    combined.paste(segmented_pil, (piece_pil.width + 20, 0))
    
    return combined

def encode_image_to_base64(image_pil):
    """Convert PIL image to base64"""
    buffer = io.BytesIO()
    image_pil.save(buffer, format='JPEG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def validate_match_with_colors(piece_image, seg_id, panoptic_mask, segmented_image):
    """Validate OpenAI match using color similarity as additional check"""
    try:
        # Get the segment color
        segment_mask = (panoptic_mask == seg_id)
        if not np.any(segment_mask):
            return 0.0
        
        # Calculate average color of the piece (excluding black background)
        piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)
        piece_mask = piece_gray > 10  # Exclude near-black pixels
        
        if not np.any(piece_mask):
            return 0.0
        
        piece_pixels = piece_image[piece_mask]
        piece_avg_color = np.mean(piece_pixels, axis=0)
        
        # Get segment pixels
        segment_pixels = segmented_image[segment_mask]
        if len(segment_pixels) == 0:
            return 0.0
        
        segment_avg_color_rgb = np.mean(segment_pixels, axis=0)
        # Convert to BGR for comparison
        segment_avg_color = [segment_avg_color_rgb[2], segment_avg_color_rgb[1], segment_avg_color_rgb[0]]
        
        # Calculate color distance (lower is better)
        color_dist = np.linalg.norm(piece_avg_color - segment_avg_color)
        
        # Convert to similarity score (0-1, higher is better)
        max_dist = 255 * np.sqrt(3)  # Maximum possible distance
        similarity = 1.0 - (color_dist / max_dist)
        
        return max(0.0, similarity)
        
    except Exception as e:
        print(f"Error in color validation: {e}")
        return 0.0

def find_best_match_combined(piece_image, segments_data, panoptic_mask, segmented_image, 
                           openai_client, panoptic_output_path, full_rgb_image):
    """Enhanced matching with color, texture, and OpenAI analysis"""
    
    # First, get color candidates
    color_candidates = []
    for seg_info in segments_data:
        seg_id = seg_info["seg_id"]
        color_similarity = validate_match_with_colors(piece_image, seg_id, panoptic_mask, segmented_image)
        color_candidates.append((seg_id, color_similarity, seg_info["class_name"]))
    
    # Calculate texture similarities
    print("Calculating texture similarities...")
    texture_scores = {}
    for seg_info in segments_data:
        seg_id = seg_info["seg_id"]
        texture_scores[seg_id] = texture_similarity(piece_image, seg_id, panoptic_mask, full_rgb_image)
    
    # Define combined scoring function
    def combined_score(seg_id):
        color_sim = next((x[1] for x in color_candidates if x[0] == seg_id), 0.0)
        texture_sim = texture_scores.get(seg_id, 0.0)
        # Weight: 60% color, 40% texture - tune as needed
        return 0.6 * color_sim + 0.4 * texture_sim
    
    # Sort by combined score
    color_candidates.sort(key=lambda x: combined_score(x[0]), reverse=True)
    
    # Get top candidates for OpenAI analysis
    top_candidates = color_candidates[:3]
    candidates_text = "Most likely candidates based on color and texture analysis:\n"
    for seg_id, color_sim, class_name in top_candidates:
        texture_sim = texture_scores[seg_id]
        combined_sim = combined_score(seg_id)
        candidates_text += f"- Segment {seg_id}: {class_name} (color: {color_sim:.2f}, texture: {texture_sim:.2f}, combined: {combined_sim:.2f})\n"
    
    print(f"Top candidates:\n{candidates_text}")
    
    # Try OpenAI with enhanced confidence threshold
    combined_image = create_side_by_side_image(piece_image, panoptic_output_path)
    openai_seg_id = ask_openai_focused(combined_image, segments_data, openai_client, candidates_text)
    
    if openai_seg_id is None:
        print("OpenAI failed, using best combined score match")
        return top_candidates[0][0] if top_candidates else None
    
    # Enhanced validation with combined score
    openai_combined_score = combined_score(openai_seg_id)
    best_combined_score = combined_score(top_candidates[0][0]) if top_candidates else 0.0
    
    print(f"OpenAI suggested segment {openai_seg_id}, combined score: {openai_combined_score:.3f}")
    print(f"Best candidate segment {top_candidates[0][0]}, combined score: {best_combined_score:.3f}")
    
    # Enhanced decision logic: use stricter thresholds
    min_acceptable_score = 0.25  # Minimum combined score to accept
    score_gap_threshold = 0.15   # Maximum acceptable gap between OpenAI and best candidate
    
    if openai_combined_score < min_acceptable_score:
        print(f"OpenAI suggestion score too low ({openai_combined_score:.3f} < {min_acceptable_score})")
        if best_combined_score > min_acceptable_score:
            print(f"Using best candidate: segment {top_candidates[0][0]} (score: {best_combined_score:.3f})")
            return top_candidates[0][0]
        else:
            print("No candidates meet minimum score threshold")
            return None
    
    if best_combined_score - openai_combined_score > score_gap_threshold:
        print(f"Score gap too large ({best_combined_score - openai_combined_score:.3f} > {score_gap_threshold})")
        print(f"Using best candidate: segment {top_candidates[0][0]}")
        return top_candidates[0][0]
    
    return openai_seg_id

def ask_openai_focused(combined_image, segments_data, openai_client, candidates_text):
    """OpenAI with focus on likely candidates"""
    
    image_base64 = encode_image_to_base64(combined_image)
    
    # Create detailed segments description
    segments_text = ""
    for seg in segments_data:
        segments_text += f"Segment {seg['seg_id']}: {seg['class_name']}\n"
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""You are an expert puzzle solver. Analyze this image carefully:

LEFT SIDE: A puzzle piece that needs classification
RIGHT SIDE: Complete puzzle with colored regions

All available segments:
{segments_text}

{candidates_text}

TASK: Determine which segment the puzzle piece belongs to by analyzing the VISUAL CONTENT:

1. What do you see in the puzzle piece? (water, rocks, trees, sky, etc.)
2. Look at the colored regions on the right - which one contains the same type of content?
3. Consider the suggested candidates above, but make your decision based on visual content match

IMPORTANT: 
- If the piece shows water/sea → choose the water/sea segment
- If the piece shows rocks/cliffs → choose the rock/terrain segment  
- If the piece shows trees/plants → choose the vegetation segment
- If the piece shows sky → choose the sky segment

Only return a segment number if you are confident in the match. If unsure, return "UNCERTAIN".

Answer: [number only or UNCERTAIN]"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=20,
            temperature=0.1
        )
        
        # Extract number from response
        answer = response.choices[0].message.content.strip().upper()
        
        if "UNCERTAIN" in answer:
            return None
            
        import re
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return int(numbers[0])
        else:
            return None
            
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def find_best_color_match(piece_image, segments_data, panoptic_mask, segmented_image):
    """Fallback method using only color matching"""
    best_seg_id = None
    best_similarity = 0.0
    
    for seg_info in segments_data:
        seg_id = seg_info["seg_id"]
        similarity = validate_match_with_colors(piece_image, seg_id, panoptic_mask, segmented_image)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_seg_id = seg_id
    
    return best_seg_id

def get_segment_color_from_panoptic(seg_id, panoptic_mask, segmented_image):
    """
    Return the exact BGR color used for the given seg_id in the panoptic segmented image.
    """
    segment_mask = (panoptic_mask == seg_id)
    if np.any(segment_mask):
        # Find the most common color among the segment's pixels
        pixels = segmented_image[segment_mask]
        most_common_bgr = Counter(map(tuple, pixels)).most_common(1)[0][0]
        return tuple(int(c) for c in most_common_bgr)

    # Fallback: deterministic pseudo-random color
    np.random.seed(seg_id)
    return tuple(int(c) for c in np.random.randint(50, 255, size=3))

def process_puzzle_matching(pieces_image, contours, panoptic_data, puzzle_original_path, 
                          openai_client, session_id, results_folder):
    """Enhanced puzzle piece matching with texture analysis"""
    try:
        print("Starting enhanced puzzle matching process...")
        
        # Load segmented image - prefer the saved numpy array over the file
        panoptic_output_path = os.path.join(results_folder, "panoptic_output.png")
        
        segments_data = panoptic_data['segments_data']
        panoptic_mask = panoptic_data['panoptic_mask']
        
        # Use the saved segmented image if available, otherwise load from file
        if 'segmented_image' in panoptic_data and panoptic_data['segmented_image'] is not None:
            segmented_image = panoptic_data['segmented_image']
            print("Using saved segmented image from memory")
        else:
            # Fallback to loading from file (convert BGR to RGB since cv2.imread loads as BGR)
            segmented_image_bgr = cv2.imread(panoptic_output_path)
            segmented_image = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2RGB)
            print("Loading segmented image from file and converting BGR->RGB")
        
        # Load original puzzle image for texture analysis
        original_bgr = cv2.imread(puzzle_original_path)
        if original_bgr is None:
            print(f"Warning: Could not load original puzzle from {puzzle_original_path}")
            # Use pieces image as fallback
            full_rgb_image = cv2.cvtColor(pieces_image, cv2.COLOR_BGR2RGB)
        else:
            full_rgb_image = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"Found {len(segments_data)} segments")
        print(f"Processing {len(contours)} pieces")
        
        # Result image
        result_image = pieces_image.copy()
        matches = {}
        match_confidences = {}
        
        # Process each piece
        for i, contour in enumerate(contours):
            print(f"\nProcessing piece {i+1}/{len(contours)}...")
            
            # Extract piece image
            x, y, w, h = cv2.boundingRect(contour)
            
            # Create mask for this piece
            piece_mask = np.zeros(pieces_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(piece_mask, [contour], -1, 255, -1)
            
            # Extract piece with some margin
            margin = 10
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(pieces_image.shape[1], x + w + margin)
            y_end = min(pieces_image.shape[0], y + h + margin)
            
            piece_region = pieces_image[y_start:y_end, x_start:x_end].copy()
            piece_mask_crop = piece_mask[y_start:y_end, x_start:x_end]
            
            # Apply mask to get clean piece
            piece_image = cv2.bitwise_and(piece_region, piece_region, mask=piece_mask_crop)
            
            # Skip very small pieces
            if piece_image.shape[0] < 30 or piece_image.shape[1] < 30:
                print(f"Skipping piece {i+1} - too small")
                continue
            
            # Get best match using enhanced combined method
            seg_id = find_best_match_combined(
                piece_image, 
                segments_data, 
                panoptic_mask, 
                segmented_image, 
                openai_client, 
                panoptic_output_path,
                full_rgb_image
            )
            
            if seg_id is not None:
                matches[i+1] = seg_id
                
                # Calculate confidence metrics for reporting
                color_conf = validate_match_with_colors(piece_image, seg_id, panoptic_mask, segmented_image)
                texture_conf = texture_similarity(piece_image, seg_id, panoptic_mask, full_rgb_image)
                combined_conf = 0.6 * color_conf + 0.4 * texture_conf
                match_confidences[i+1] = {
                    'color': color_conf,
                    'texture': texture_conf,
                    'combined': combined_conf
                }
                
                # Find segment info
                seg_info = next((s for s in segments_data if s["seg_id"] == seg_id), None)
                if seg_info:
                    print(f"Piece {i+1} -> Segment {seg_id} ({seg_info['class_name']}) [conf: {combined_conf:.3f}]")
                else:
                    print(f"Piece {i+1} -> Segment {seg_id} [conf: {combined_conf:.3f}]")
                
                # Color the piece with the EXACT color from panoptic segmentation
                color_bgr = get_segment_color_from_panoptic(seg_id, panoptic_mask, segmented_image)
                
                # Fill the piece with the segment color
                cv2.fillPoly(result_image, [contour], color_bgr)
                
                # Add black border
                cv2.drawContours(result_image, [contour], -1, (0, 0, 0), 2)
                
                # Add label
                label_text = seg_info['class_name'][:8] if seg_info else f"Seg{seg_id}"
                cv2.putText(result_image, label_text, (x + w//2 - 20, y + h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            else:
                print(f"No confident match found for piece {i+1}")
                # Draw white border for unmatched pieces
                cv2.drawContours(result_image, [contour], -1, (255, 255, 255), 2)
        
        print(f"\nMatching complete! Found matches for {len(matches)}/{len(contours)} pieces")
        
        # Print confidence statistics
        if match_confidences:
            avg_color = np.mean([c['color'] for c in match_confidences.values()])
            avg_texture = np.mean([c['texture'] for c in match_confidences.values()])
            avg_combined = np.mean([c['combined'] for c in match_confidences.values()])
            print(f"Average confidences - Color: {avg_color:.3f}, Texture: {avg_texture:.3f}, Combined: {avg_combined:.3f}")
        
        # Save matches info with confidence scores
        matches_file = os.path.join(results_folder, f"matches_{session_id}.json")
        match_data = {
            'matches': matches,
            'confidences': match_confidences,
            'summary': {
                'total_pieces': len(contours),
                'matched_pieces': len(matches),
                'match_rate': len(matches) / len(contours) if contours else 0
            }
        }
        
        with open(matches_file, 'w') as f:
            json.dump(match_data, f, indent=2)
        
        return result_image
        
    except Exception as e:
        print(f"Error in enhanced puzzle matching: {e}")
        raise

# Legacy functions for backward compatibility
def create_openai_area_visualization(pieces_img, contours, panoptic_data, panoptic_output_path, openai_client, puzzle_original_path):
    """Legacy function - use new system"""
    print("Using legacy function - redirecting to enhanced system")
    return process_puzzle_matching(pieces_img, contours, panoptic_data, puzzle_original_path, openai_client, "legacy", ".")

def create_traditional_area_visualization(pieces_img, contours, panoptic_data, panoptic_output_path, puzzle_original_path):
    """Legacy function - basic fallback"""
    print("Using legacy traditional function")
    result = pieces_img.copy()
    
    # Just draw white borders on all pieces
    for contour in contours:
        cv2.drawContours(result, [contour], -1, (255, 255, 255), 2)
    
    return result