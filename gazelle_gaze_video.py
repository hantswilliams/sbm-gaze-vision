# -*- coding: utf-8 -*-
"""gazelle-gaze-video.py

Video gaze analysis using Gazelle model with memory management and data export.

This script provides functionality for:
1. Processing videos to visualize gaze direction
2. Exporting detailed per-frame data (JSON/CSV)
3. Generating summary statistics for gaze behavior
4. Managing memory for efficient processing

Installation:
    pip install torch torchvision timm scikit-learn matplotlib pandas opencv-python tqdm pillow numpy mediapipe psutil
    pip install retina-face

Usage:
    from gazelle_gaze_video import VideoGazeAnalyzer
    
    # Initialize analyzer
    analyzer = VideoGazeAnalyzer()
    
    # Basic processing
    analyzer.process_video('input.mp4', 'output.mp4')
    
    # Advanced processing with data export
    analyzer.process_video_with_data_export('input.mp4', 'output_with_data.mp4')

Troubleshooting:
    1. "RuntimeError: stack expects a non-empty TensorList":
       This occurs when the Gazelle model can't find valid faces in a frame.
       The script now handles this error gracefully and continues processing.
    
    2. No output files generated:
       Ensure your video contains detectable faces. The script creates output
       files even when some frames have no faces, but at least one frame
       needs to have a face for complete data export.
    
    3. Memory errors:
       Use the memory management functions (clear_memory, cleanup_completely)
       to free up memory during and after processing.
"""

# Install required packages (run once)
# pip install torch torchvision timm scikit-learn matplotlib pandas opencv-python tqdm pillow numpy mediapipe psutil
# pip install retina-face

# !pip install retina-face

import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from retinaface import RetinaFace
from tqdm import tqdm
import os
import gc
import psutil
import json
import pandas as pd
from datetime import datetime
import math

# Standalone Memory Management Functions
def clear_gpu_memory():
    """
    Clear GPU memory and force garbage collection.
    Call this between processing different videos.
    """
    print("Clearing GPU memory...")
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Print GPU memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
    
    # Clear MPS cache if using Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("MPS cache cleared")
    
    # Force garbage collection
    gc.collect()
    
    # Print system memory usage
    memory = psutil.virtual_memory()
    print(f"System Memory - Used: {memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB ({memory.percent:.1f}%)")
    print("Memory cleanup complete!\n")

def cleanup_model_and_memory(model=None, analyzer=None):
    """
    Complete cleanup: remove model from memory and clear GPU cache.
    Use this when switching to a completely different model or finishing work.
    
    Args:
        model: PyTorch model to move to CPU and delete
        analyzer: VideoGazeAnalyzer instance to cleanup
    """
    print("Performing complete cleanup...")
    
    if analyzer is not None:
        # Use the analyzer's cleanup method
        analyzer.cleanup_completely()
    elif model is not None:
        # Move model to CPU and delete
        model.cpu()
        del model
        clear_gpu_memory()
    else:
        # Just clear memory
        clear_gpu_memory()
    
    print("Complete cleanup finished!")

class VideoGazeAnalyzer:
    def __init__(self, use_cuda=True):
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        try:
            # Load Gazelle model
            self.model, self.transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
            self.model.eval()
            self.model.to(self.device)
            print("Gazelle model loaded successfully")
        except Exception as e:
            print(f"Error loading Gazelle model: {e}")
            print("Check that you have an internet connection and that the model repository is accessible.")
            raise

        # Colors for visualization
        self.colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']

    def process_frame(self, frame):
        """Process a single frame and return the visualization"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        width, height = image.size

        # Detect faces
        resp = RetinaFace.detect_faces(frame_rgb)
        if not isinstance(resp, dict):
            return frame  # Return original frame if no faces detected

        # Extract bounding boxes
        bboxes = [resp[key]['facial_area'] for key in resp.keys()]
        norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height])
                       for bbox in bboxes]]

        # Prepare input for Gazelle
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        input_data = {
            "images": img_tensor,
            "bboxes": norm_bboxes
        }

        # Get model predictions
        with torch.no_grad():
            output = self.model(input_data)

        # Visualize results
        result_image = self.visualize_all(
            image,
            output['heatmap'][0],
            norm_bboxes[0],
            output['inout'][0] if output['inout'] is not None else None
        )

        # Convert back to BGR for OpenCV
        result_array = np.array(result_image)
        return cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

    def visualize_all(self, pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
        """Visualize all detected faces and their gaze directions"""
        overlay_image = pil_image.convert("RGBA")
        draw = ImageDraw.Draw(overlay_image)
        width, height = pil_image.size

        for i in range(len(bboxes)):
            bbox = bboxes[i]
            xmin, ymin, xmax, ymax = bbox
            color = self.colors[i % len(self.colors)]

            # Draw face bounding box
            draw.rectangle(
                [xmin * width, ymin * height, xmax * width, ymax * height],
                outline=color,
                width=int(min(width, height) * 0.01)
            )

            if inout_scores is not None:
                inout_score = inout_scores[i]

                # Draw in-frame score
                text = f"in-frame: {inout_score:.2f}"
                text_y = ymax * height + int(height * 0.01)
                draw.text(
                    (xmin * width, text_y),
                    text,
                    fill=color,
                    font=None  # Using default font
                )

                # Draw gaze direction if looking in-frame
                if inout_score > inout_thresh:
                    heatmap = heatmaps[i]
                    heatmap_np = heatmap.detach().cpu().numpy()
                    max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)

                    # Calculate gaze target and face center
                    gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
                    gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
                    bbox_center_x = ((xmin + xmax) / 2) * width
                    bbox_center_y = ((ymin + ymax) / 2) * height

                    # Draw gaze target point and line
                    draw.ellipse(
                        [(gaze_target_x-5, gaze_target_y-5),
                         (gaze_target_x+5, gaze_target_y+5)],
                        fill=color,
                        width=int(0.005*min(width, height))
                    )
                    draw.line(
                        [(bbox_center_x, bbox_center_y),
                         (gaze_target_x, gaze_target_y)],
                        fill=color,
                        width=int(0.005*min(width, height))
                    )

        # Convert to RGB for OpenCV compatibility
        return overlay_image.convert('RGB')

    def process_video(self, input_path, output_path, start_time=0, duration=None):
        """Process a video file and save the result"""
        # Open video file
        cap = cv2.VideoCapture(input_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate start and end frames
        start_frame = int(start_time * fps)
        if duration:
            end_frame = start_frame + int(duration * fps)
        else:
            end_frame = total_frames

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            with tqdm(total=end_frame-start_frame) as pbar:
                frame_count = start_frame
                while cap.isOpened() and frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Process frame
                    processed_frame = self.process_frame(frame)
                    out.write(processed_frame)

                    frame_count += 1
                    pbar.update(1)

        finally:
            # Clean up
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Clear memory after video processing
            self.clear_memory()

    def clear_memory(self):
        """
        Clear GPU/MPS memory and force garbage collection.
        Call this between processing different videos to free up memory.
        """
        print("Clearing memory...")
        
        # Clear PyTorch CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Print GPU memory usage
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
        
        # Clear MPS cache if using Apple Silicon
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("MPS cache cleared")
        
        # Force garbage collection
        gc.collect()
        
        # Print system memory usage
        memory = psutil.virtual_memory()
        print(f"System Memory - Used: {memory.used / 1024**3:.2f} GB / {memory.total / 1024**3:.2f} GB ({memory.percent:.1f}%)")
        print("Memory cleanup complete!\n")
    
    def cleanup_completely(self):
        """
        Complete cleanup: move model to CPU and clear all memory.
        Use this when you're done with the analyzer or want to free maximum memory.
        """
        print("Performing complete cleanup...")
        
        # Move model to CPU to free GPU memory
        if hasattr(self, 'model'):
            self.model.cpu()
            print("Model moved to CPU")
        
        # Clear memory
        self.clear_memory()
        
        print("Complete cleanup finished!")
    
    def reinitialize_model(self, use_cuda=True):
        """
        Reinitialize the model (useful after complete cleanup).
        """
        print("Reinitializing model...")
        
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Reload model if needed
        if not hasattr(self, 'model') or self.model is None:
            self.model, self.transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
        
        self.model.eval()
        self.model.to(self.device)
        
        print("Model reinitialized!")

    def process_frame_with_data(self, frame, frame_number=None):
        """
        Process a single frame and return the visualization along with gaze data
        
        Args:
            frame: OpenCV BGR image
            frame_number: Optional frame number for data tracking
            
        Returns:
            processed_frame: Processed OpenCV BGR image
            frame_data: Dictionary with gaze metrics for the frame
        """
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            width, height = image.size
            
            # Initialize frame data
            frame_data = {
                "frame_number": frame_number,
                "timestamp": datetime.now().isoformat(),
                "faces": []
            }
            
            # Detect faces
            resp = RetinaFace.detect_faces(frame_rgb)
            if not isinstance(resp, dict) or len(resp) == 0:
                print(f"No faces detected in frame {frame_number}")
                return frame, frame_data  # Return original frame if no faces detected
                
            # Debug: Print the first face landmarks structure if this is the first frame
            if frame_number == 0 or frame_number is None:
                if len(resp) > 0:
                    first_face_key = list(resp.keys())[0]
                    print(f"Landmark structure for first face: {resp[first_face_key]['landmarks'].keys()}")
                    
            # Extract bounding boxes
            bboxes = [resp[key]['facial_area'] for key in resp.keys()]
            landmarks = [resp[key]['landmarks'] for key in resp.keys()]
            
            # Check if there are any detected faces
            if len(bboxes) == 0:
                print(f"No valid bounding boxes in frame {frame_number}")
                return frame, frame_data  # Return original frame if no faces detected
                
            norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height])
                           for bbox in bboxes]]
            
            # Prepare input for Gazelle
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            input_data = {
                "images": img_tensor,
                "bboxes": norm_bboxes
            }
            
            try:
                # Get model predictions
                with torch.no_grad():
                    output = self.model(input_data)
            except RuntimeError as e:
                if "stack expects a non-empty TensorList" in str(e):
                    print(f"Error in Gazelle model: {e} - This typically happens when no valid faces are detected.")
                    return frame, frame_data
                else:
                    raise  # Re-raise if it's a different RuntimeError
            
            # Collect data for each face
            for i in range(len(bboxes)):
                face_data = {}
                
                # Add bounding box info
                bbox = bboxes[i]
                face_data["bbox"] = {
                    "xmin": int(bbox[0]),
                    "ymin": int(bbox[1]),
                    "xmax": int(bbox[2]),
                    "ymax": int(bbox[3]),
                    "width": int(bbox[2] - bbox[0]),
                    "height": int(bbox[3] - bbox[1])
                }
                
                # Add face center
                face_data["face_center"] = {
                    "x": int((bbox[0] + bbox[2]) / 2),
                    "y": int((bbox[1] + bbox[3]) / 2)
                }
                
                # Add landmarks with fallback for different key naming conventions
                landmark = landmarks[i]
                try:
                    face_data["landmarks"] = {
                        "left_eye": [int(landmark["left_eye"][0]), int(landmark["left_eye"][1])],
                        "right_eye": [int(landmark["right_eye"][0]), int(landmark["right_eye"][1])],
                        "nose": [int(landmark["nose"][0]), int(landmark["nose"][1])],
                        "mouth_left": [int(landmark["mouth_left"][0]), int(landmark["mouth_left"][1])],
                        "mouth_right": [int(landmark["mouth_right"][0]), int(landmark["mouth_right"][1])]
                    }
                except KeyError as e:
                    print(f"Warning: Landmark key error - {e}. Attempting fallback mapping...")
                    # Create a mapping dictionary for possible key variations
                    key_mapping = {
                        "left_eye": ["left_eye", "eye_left"],
                        "right_eye": ["right_eye", "eye_right"],
                        "nose": ["nose"],
                        "mouth_left": ["mouth_left", "left_mouth"],
                        "mouth_right": ["mouth_right", "right_mouth"]
                    }
                    
                    # Try to map the keys
                    landmarks_dict = {}
                    for target_key, possible_keys in key_mapping.items():
                        for possible_key in possible_keys:
                            if possible_key in landmark:
                                landmarks_dict[target_key] = [int(landmark[possible_key][0]), int(landmark[possible_key][1])]
                                break
                        
                        # If we couldn't find a mapping for this key, set a placeholder
                        if target_key not in landmarks_dict:
                            print(f"Warning: Could not find a mapping for {target_key}")
                            # Use face center as fallback
                            landmarks_dict[target_key] = [
                                face_data["face_center"]["x"],
                                face_data["face_center"]["y"]
                            ]
                    face_data["landmarks"] = landmarks_dict
                
                # Add in-out scores
                if output['inout'] is not None:
                    inout_score = float(output['inout'][0][i].item())
                    face_data["inout_score"] = inout_score
                    
                    # Add gaze target if looking in-frame
                    if inout_score > 0.5:  # Using 0.5 as threshold
                        heatmap = output['heatmap'][0][i]
                        heatmap_np = heatmap.detach().cpu().numpy()
                        
                        # Get max heatmap location
                        max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                        gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
                        gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
                        
                        face_data["gaze_target"] = {
                            "x": int(gaze_target_x),
                            "y": int(gaze_target_y)
                        }
                        
                        # Calculate gaze vector
                        bbox_center_x = face_data["face_center"]["x"]
                        bbox_center_y = face_data["face_center"]["y"]
                        
                        # Vector components
                        vector_x = gaze_target_x - bbox_center_x
                        vector_y = gaze_target_y - bbox_center_y
                        
                        # Vector length (Euclidean distance)
                        distance = math.sqrt(vector_x**2 + vector_y**2)
                        
                        # Normalize vector
                        if distance > 0:
                            norm_vector_x = vector_x / distance
                            norm_vector_y = vector_y / distance
                        else:
                            norm_vector_x = 0
                            norm_vector_y = 0
                        
                        # Angle in degrees
                        angle_rad = math.atan2(vector_y, vector_x)
                        angle_deg = math.degrees(angle_rad)
                        
                        face_data["gaze_vector"] = {
                            "x": float(vector_x),
                            "y": float(vector_y),
                            "normalized_x": float(norm_vector_x),
                            "normalized_y": float(norm_vector_y),
                            "distance": float(distance),
                            "angle_degrees": float(angle_deg)
                        }
                        
                        # Add heatmap statistics
                        face_data["heatmap_stats"] = {
                            "max_value": float(np.max(heatmap_np)),
                            "mean_value": float(np.mean(heatmap_np)),
                            "std_value": float(np.std(heatmap_np))
                        }
                
                # Add face data to frame
                frame_data["faces"].append(face_data)
            
            # Visualize results
            result_image = self.visualize_all(
                image,
                output['heatmap'][0],
                norm_bboxes[0],
                output['inout'][0] if output['inout'] is not None else None
            )
            
            # Convert back to BGR for OpenCV
            result_array = np.array(result_image)
            processed_frame = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
            
            return processed_frame, frame_data
        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            return frame, {"error": str(e)}

    def process_video_with_data_export(self, input_path, output_path, start_time=0, duration=None):
        """
        Process a video file, save the result, and export frame-by-frame gaze data
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (will be created)
            start_time: Time in seconds to start processing from
            duration: Duration in seconds to process (None for full video)
            
        Returns:
            output_folder: Path to folder containing exported data
        """
        # Create output folder based on output video name
        output_base = os.path.splitext(output_path)[0]
        output_folder = f"{output_base}_data"
        os.makedirs(output_folder, exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' does not exist.")
            # Create empty data files with error message
            error_data = [{
                "error": f"Input file '{input_path}' does not exist.",
                "timestamp": datetime.now().isoformat(),
                "faces": []
            }]
            self.save_frame_data(error_data, output_folder)
            self.generate_summary_stats(error_data, output_folder)
            return output_folder
            
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{input_path}'.")
            # Create empty data files with error message
            error_data = [{
                "error": f"Could not open video file '{input_path}'.",
                "timestamp": datetime.now().isoformat(),
                "faces": []
            }]
            self.save_frame_data(error_data, output_folder)
            self.generate_summary_stats(error_data, output_folder)
            return output_folder

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate start and end frames
        start_frame = int(start_time * fps)
        if duration:
            end_frame = start_frame + int(duration * fps)
        else:
            end_frame = total_frames

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (frame_width, frame_height)
        )
        
        # Prepare for data collection
        all_frame_data = []
        
        # Track processing stats
        total_processed = 0
        total_with_faces = 0
        total_errors = 0

        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        try:
            with tqdm(total=end_frame-start_frame) as pbar:
                frame_count = start_frame
                while cap.isOpened() and frame_count < end_frame:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    try:
                        # Process frame and get data
                        processed_frame, frame_data = self.process_frame_with_data(frame, frame_number=frame_count)
                        
                        # Add timestamp in seconds
                        frame_data["timestamp_seconds"] = (frame_count - start_frame) / fps
                        
                        # Write processed frame to video
                        out.write(processed_frame)
                        
                        # Update stats
                        total_processed += 1
                        if len(frame_data["faces"]) > 0:
                            total_with_faces += 1
                            
                        # Store frame data
                        all_frame_data.append(frame_data)
                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")
                        # Write original frame to video on error
                        out.write(frame)
                        # Add empty frame data
                        empty_frame_data = {
                            "frame_number": frame_count,
                            "timestamp": datetime.now().isoformat(),
                            "timestamp_seconds": (frame_count - start_frame) / fps,
                            "faces": [],
                            "error": str(e)
                        }
                        all_frame_data.append(empty_frame_data)
                        total_errors += 1

                    frame_count += 1
                    pbar.update(1)
                    
            # Print processing summary
            print(f"\nProcessing summary:")
            print(f"- Total frames processed: {total_processed}")
            print(f"- Frames with faces detected: {total_with_faces}")
            print(f"- Frames with errors: {total_errors}")
            print(f"- Face detection rate: {total_with_faces/total_processed*100:.1f}% of processed frames")
                    
            # Save all collected data (even if empty - will be handled in save_frame_data)
            self.save_frame_data(all_frame_data, output_folder)
            
            # Generate summary statistics
            self.generate_summary_stats(all_frame_data, output_folder)
            
            print(f"Video processing complete. Data exported to {output_folder}")
            return output_folder

        except Exception as e:
            print(f"Error during video processing: {e}")
            # Save whatever data we collected so far
            if all_frame_data:
                self.save_frame_data(all_frame_data, output_folder)
                self.generate_summary_stats(all_frame_data, output_folder)
            else:
                # Create empty data files with error message
                error_data = [{
                    "error": f"Processing failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                    "faces": []
                }]
                self.save_frame_data(error_data, output_folder)
                self.generate_summary_stats(error_data, output_folder)
            
            return output_folder
        finally:
            # Clean up
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Clear memory after video processing
            self.clear_memory()
            
    def save_frame_data(self, all_frame_data, output_folder):
        """Save frame data to JSON and CSV files"""
        if not all_frame_data:
            print("Warning: No frame data to save")
            empty_data = [{
                "frame_number": 0,
                "timestamp": datetime.now().isoformat(),
                "timestamp_seconds": 0,
                "faces": [],
                "warning": "No valid frames with faces were processed"
            }]
            all_frame_data = empty_data
            
        # Save raw JSON data
        json_path = os.path.join(output_folder, "gaze_data.json")
        with open(json_path, 'w') as f:
            json.dump(all_frame_data, f, indent=2)
            
        print(f"Saved detailed frame data to {json_path}")
        
        # Flatten data for CSV export
        flat_data = []
        
        for frame in all_frame_data:
            frame_number = frame.get("frame_number", None)
            timestamp_seconds = frame.get("timestamp_seconds", None)
            
            # If no faces in this frame, add a row with just the frame info
            if len(frame["faces"]) == 0:
                flat_row = {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "has_face": False
                }
                # Add any error message if present
                if "error" in frame:
                    flat_row["error"] = frame["error"]
                if "warning" in frame:
                    flat_row["warning"] = frame["warning"]
                    
                flat_data.append(flat_row)
            
            # Otherwise add a row for each face
            for face_idx, face in enumerate(frame["faces"]):
                flat_row = {
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "has_face": True,
                    "face_index": face_idx
                }
                
                # Add face bounding box
                if "bbox" in face:
                    for key, value in face["bbox"].items():
                        flat_row[f"bbox_{key}"] = value
                
                # Add face center
                if "face_center" in face:
                    for key, value in face["face_center"].items():
                        flat_row[f"face_center_{key}"] = value
                
                # Add landmarks (flatten the structure)
                if "landmarks" in face:
                    for landmark_name, coords in face["landmarks"].items():
                        flat_row[f"landmark_{landmark_name}_x"] = coords[0]
                        flat_row[f"landmark_{landmark_name}_y"] = coords[1]
                
                # Add inout score
                if "inout_score" in face:
                    flat_row["inout_score"] = face["inout_score"]
                
                # Add gaze target
                if "gaze_target" in face:
                    for key, value in face["gaze_target"].items():
                        flat_row[f"gaze_target_{key}"] = value
                
                # Add gaze vector
                if "gaze_vector" in face:
                    for key, value in face["gaze_vector"].items():
                        flat_row[f"gaze_vector_{key}"] = value
                
                # Add heatmap stats
                if "heatmap_stats" in face:
                    for key, value in face["heatmap_stats"].items():
                        flat_row[f"heatmap_{key}"] = value
                
                flat_data.append(flat_row)
        
        # Save as CSV
        csv_path = os.path.join(output_folder, "gaze_data.csv")
        df = pd.DataFrame(flat_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Saved flattened frame data to {csv_path}")
    
    def generate_summary_stats(self, all_frame_data, output_folder):
        """Generate and save summary statistics from all frame data"""
        # Initialize counters and accumulators
        total_frames = len(all_frame_data)
        frames_with_faces = 0
        total_faces = 0
        faces_looking_in_frame = 0
        
        # For averaging
        all_inout_scores = []
        all_gaze_distances = []
        all_gaze_angles = []
        
        # Count frames with error messages
        frames_with_errors = sum(1 for frame in all_frame_data if "error" in frame)
        
        for frame in all_frame_data:
            if len(frame["faces"]) > 0:
                frames_with_faces += 1
                total_faces += len(frame["faces"])
                
                for face in frame["faces"]:
                    if "inout_score" in face:
                        all_inout_scores.append(face["inout_score"])
                        
                        if face["inout_score"] > 0.5:  # Using 0.5 as threshold
                            faces_looking_in_frame += 1
                            
                            if "gaze_vector" in face:
                                all_gaze_distances.append(face["gaze_vector"]["distance"])
                                all_gaze_angles.append(face["gaze_vector"]["angle_degrees"])
        
        # Calculate summary statistics
        summary = {
            "video_stats": {
                "total_frames": total_frames,
                "frames_with_faces": frames_with_faces,
                "frames_with_errors": frames_with_errors,
                "face_detection_rate": frames_with_faces / total_frames if total_frames > 0 else 0
            },
            "face_stats": {
                "total_faces_detected": total_faces,
                "average_faces_per_frame": total_faces / total_frames if total_frames > 0 else 0
            }
        }
        
        # Only add gaze stats if we have faces
        if total_faces > 0:
            summary["gaze_stats"] = {
                "faces_looking_in_frame": faces_looking_in_frame,
                "in_frame_percentage": faces_looking_in_frame / total_faces if total_faces > 0 else 0,
                "average_inout_score": float(np.mean(all_inout_scores)) if all_inout_scores else None,
                "average_gaze_distance": float(np.mean(all_gaze_distances)) if all_gaze_distances else None,
                "average_gaze_angle": float(np.mean(all_gaze_angles)) if all_gaze_angles else None
            }
        else:
            summary["gaze_stats"] = {
                "warning": "No faces detected in any frame, gaze statistics unavailable"
            }
        
        # Save summary statistics
        summary_path = os.path.join(output_folder, "summary_stats.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Saved summary statistics to {summary_path}")
        
        return summary

# Example usage
if __name__ == "__main__":
    # Memory Management Usage Examples
    
    print("=== Gazelle Gaze Video Analysis ===")
    
    # Example 1: Basic video processing
    print("\n=== Example 1: Basic Video Processing ===")
    
    # Initialize analyzer
    analyzer = VideoGazeAnalyzer(use_cuda=True)
    
    # Process video with basic visualization
    input_video = "/content/video.mp4"  # Replace with your video path
    output_video = "output_basic.mp4"
    
    print("Processing video with basic visualization...")
    # analyzer.process_video(input_video, output_video)
    print("Basic processing example ready - uncomment the line above to run")
    
    # Clear memory between processing tasks
    analyzer.clear_memory()
    
    # Example 2: Advanced video processing with data export
    print("\n=== Example 2: Video Processing with Data Export ===")
    
    output_video_with_data = "output_with_data.mp4"
    
    print("Processing video with data export...")
    output_folder = analyzer.process_video_with_data_export(
        input_video,
        output_video_with_data,
        start_time=0,  # Start from beginning
        duration=None  # Process entire video
    )
    print(f"Data exported to {output_folder}")
    print("Data export example ready - uncomment the lines above to run")
    
    # Example 3: Processing a clip from a longer video
    print("\n=== Example 3: Processing a Video Clip ===")
    
    output_clip = "output_clip.mp4"
    
    print("Processing 10-second clip starting at 5 seconds...")
    # clip_output_folder = analyzer.process_video_with_data_export(
    #     input_video,
    #     output_clip,
    #     start_time=5,    # Start 5 seconds in
    #     duration=10      # Process 10 seconds
    # )
    # print(f"Clip data exported to {clip_output_folder}")
    print("Clip processing example ready - uncomment the lines above to run")
    
    # Complete cleanup when done
    analyzer.cleanup_completely()
    
    print("\n=== Memory Management Tips ===")
    print("1. Call analyzer.clear_memory() between videos")
    print("2. Call analyzer.cleanup_completely() when completely done")
    print("3. Use analyzer.reinitialize_model() to restart after complete cleanup")
    print("4. Use standalone functions clear_gpu_memory() and cleanup_model_and_memory() as needed")
    
    print("\n=== Data Export Information ===")
    print("When using process_video_with_data_export, the following files are created:")
    print("1. output_video_data/gaze_data.json - Detailed JSON data for all frames")
    print("2. output_video_data/gaze_data.csv - Flattened CSV data for easy analysis")
    print("3. output_video_data/summary_stats.json - Summary statistics for the video")
    
    print("\nTo use this module in your own code:")
    print("from gazelle_gaze_video import VideoGazeAnalyzer")
    print("analyzer = VideoGazeAnalyzer()")
    print("analyzer.process_video_with_data_export('input.mp4', 'output.mp4')")
