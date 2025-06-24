from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
import os
import uuid
import cv2
import shutil
import config
from openai import OpenAI
from panoptic_segmentation import panoptic_segmentation
from find_contour import detect_puzzle_pieces
from detect_corner_simple import find_square_corners, create_corners_visualization
from identify_piece_shapes import classify_all_pieces, group_pieces_by_shape, create_shape_visualization, create_visualization
from puzzle_area_match import (
    save_panoptic_data,
    load_panoptic_data,
    process_puzzle_matching,
    extract_individual_regions,
    save_individual_regions,
    create_openai_area_visualization,
    create_traditional_area_visualization
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.secret_key = 'your-secret-key-change-this'


@app.route("/")
def index():
    return redirect(url_for('welcome'))

@app.route("/welcome")
def welcome():
    return render_template("welcome.html")

@app.route("/select")
def select_method():
    # Clear results folder when entering select page
    clear_results_folder()
    # Clear session data
    session.clear()
    return render_template("select_method.html")

@app.route("/by_shape", methods=["GET", "POST"])
def by_shape():
    result_image = None
    uploaded_pieces = None
    
    if request.method == "POST":
        if "pieces_image" not in request.files:
            print("No file part")
        else:
            file = request.files["pieces_image"]
            if file and file.filename:
                # Save uploaded pieces image for display
                pieces_filename = "uploaded_pieces.jpg"
                pieces_path = os.path.join(app.config['RESULT_FOLDER'], pieces_filename)
                file.save(pieces_path)
                uploaded_pieces = pieces_filename

                # Process the image
                image = cv2.imread(pieces_path)
                pieces, contours, _ = detect_puzzle_pieces(image)
                results = classify_all_pieces(contours)
                groups = group_pieces_by_shape(results)
                
                # Create main result
                result_img, _ = create_shape_visualization(image, results, groups)
                result_name = "shape_result.jpg"
                save_path = os.path.join(app.config['RESULT_FOLDER'], result_name)
                cv2.imwrite(save_path, result_img)
                print(f"Saved main result: {save_path}")

                # Create corners visualization
                try:
                    corners_list = [find_square_corners(c) for c in contours]
                    corners_img = create_corners_visualization(image, contours, corners_list)
                    corners_save_path = os.path.join(app.config['RESULT_FOLDER'], "corners_result.jpg")
                    cv2.imwrite(corners_save_path, corners_img)
                    print(f"Saved corners result: {corners_save_path}")
                except Exception as e:
                    print(f"Error creating corners visualization: {e}")

                # Create edge classification visualization
                try:
                    edge_img = create_visualization(image, results)
                    edge_save_path = os.path.join(app.config['RESULT_FOLDER'], "edge_classification_result.jpg")
                    cv2.imwrite(edge_save_path, edge_img)
                    print(f"Saved edge classification result: {edge_save_path}")
                except Exception as e:
                    print(f"Error creating edge classification visualization: {e}")

                result_image = result_name
            else:
                print("No selected file")

    return render_template("by_shape.html", result_image=result_image, uploaded_pieces=uploaded_pieces)


                    
@app.route("/by_area", methods=["GET", "POST"])
def by_area():
    puzzle_uploaded = False
    pieces_uploaded = False
    panoptic_ready = False
    area_result_ready = False
    error_message = None
    
    # Check what files exist
    puzzle_original_path = os.path.join(app.config['RESULT_FOLDER'], "puzzle_original.png")
    panoptic_output_path = os.path.join(app.config['RESULT_FOLDER'], "panoptic_output.png")
    pieces_original_path = os.path.join(app.config['RESULT_FOLDER'], "pieces_original.png")
    pieces_attributed_path = os.path.join(app.config['RESULT_FOLDER'], "pieces_attributed.png")
    
    puzzle_uploaded = os.path.exists(puzzle_original_path)
    panoptic_ready = os.path.exists(panoptic_output_path)
    pieces_uploaded = os.path.exists(pieces_original_path)
    area_result_ready = os.path.exists(pieces_attributed_path)
    
    if request.method == "POST":
        try:
            session_id = session.get('session_id')
            if not session_id:
                session_id = str(uuid.uuid4())
                session['session_id'] = session_id
            
            # Handle puzzle image upload
            if "puzzle_image" in request.files and request.files["puzzle_image"].filename:
                puzzle_file = request.files["puzzle_image"]
                
                print("Processing puzzle image upload...")
                
                # Save original puzzle
                puzzle_path = os.path.join(app.config['UPLOAD_FOLDER'], f"puzzle_{session_id}.png")
                puzzle_file.save(puzzle_path)
                cv2.imwrite(puzzle_original_path, cv2.imread(puzzle_path))
                
                print("Running panoptic segmentation...")
                
                # Run panoptic segmentation
                result_img, segments_data, panoptic_mask = panoptic_segmentation(puzzle_path)
                cv2.imwrite(panoptic_output_path, result_img)
                
                print(f"Panoptic segmentation completed with {len(segments_data)} segments")
                
                # Save panoptic data WITH the segmented image to preserve exact colors
                save_panoptic_data(segments_data, panoptic_mask, session_id, app.config['RESULT_FOLDER'], result_img)
                
                # Extract and save individual regions
                original_puzzle = cv2.imread(puzzle_original_path)
                individual_regions = extract_individual_regions(original_puzzle, panoptic_mask, segments_data)
                saved_regions = save_individual_regions(individual_regions, session_id, app.config['RESULT_FOLDER'])
                
                print(f"Extracted and saved {len(saved_regions)} individual regions")
                
                puzzle_uploaded = True
                panoptic_ready = True
                print("Puzzle image processed successfully")
            
            # Handle pieces image upload
            if "pieces_image" in request.files and request.files["pieces_image"].filename:
                pieces_file = request.files["pieces_image"]
                
                print("Processing pieces image upload...")
                
                # Check if we have panoptic data
                panoptic_data = load_panoptic_data(session.get('session_id'), app.config['RESULT_FOLDER'])
                if not panoptic_data:
                    error_message = "Please upload the puzzle image first"
                else:
                    # Save pieces image
                    pieces_path = os.path.join(app.config['UPLOAD_FOLDER'], f"pieces_{session_id}.png")
                    pieces_file.save(pieces_path)
                    cv2.imwrite(pieces_original_path, cv2.imread(pieces_path))
                    
                    # Process pieces attribution using the new system
                    pieces_img = cv2.imread(pieces_path)
                    pieces, contours, _ = detect_puzzle_pieces(pieces_img)
                    
                    print(f"Detected {len(contours)} puzzle pieces")
                    print("Starting puzzle piece matching process...")
                    
                    try:
                        # Use the new matching system (same as your working code)
                        attributed_img = process_puzzle_matching(
                            pieces_img, 
                            contours, 
                            panoptic_data, 
                            puzzle_original_path, 
                            openai_client, 
                            session_id, 
                            app.config['RESULT_FOLDER']
                        )
                        cv2.imwrite(pieces_attributed_path, attributed_img)
                        print("Puzzle matching completed successfully!")
                        
                    except Exception as matching_error:
                        print(f"New matching system failed: {matching_error}")
                        print("Falling back to legacy method...")
                        try:
                            # Fallback to legacy OpenAI method
                            attributed_img = create_openai_area_visualization(
                                pieces_img, contours, panoptic_data, panoptic_output_path, openai_client, puzzle_original_path
                            )
                            cv2.imwrite(pieces_attributed_path, attributed_img)
                            print("Legacy OpenAI method completed successfully!")
                        except Exception as legacy_error:
                            print(f"Legacy OpenAI method failed: {legacy_error}")
                            print("Using traditional fallback method...")
                            # Final fallback to traditional method
                            attributed_img = create_traditional_area_visualization(
                                pieces_img, contours, panoptic_data, panoptic_output_path, puzzle_original_path
                            )
                            cv2.imwrite(pieces_attributed_path, attributed_img)
                            print("Traditional fallback method completed")
                    
                    pieces_uploaded = True
                    area_result_ready = True
                    print("Pieces image processed successfully")
                    
        except Exception as e:
            error_message = f"Error processing images: {str(e)}"
            print(f"Error in by_area: {e}")
    
    return render_template("by_area.html", 
                         puzzle_uploaded=puzzle_uploaded,
                         pieces_uploaded=pieces_uploaded,
                         panoptic_ready=panoptic_ready,
                         area_result_ready=area_result_ready,
                         error_message=error_message)

@app.route("/results/<n>")
def show_result(n):
    return send_from_directory(app.config['RESULT_FOLDER'], n)

def clear_results_folder():
    """Clear all files in the results folder"""
    try:
        if os.path.exists(app.config['RESULT_FOLDER']):
            shutil.rmtree(app.config['RESULT_FOLDER'])
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
        print("Results folder cleared and recreated")
    except Exception as e:
        print(f"Error clearing results folder: {e}")



if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True)

