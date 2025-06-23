import cv2
import os
from pathlib import Path
import torch
from ultralytics import YOLO
import subprocess
import shutil
from Single_cut_frames import extract_faces  # Add import for extract_faces function

def extract_frames(video_path, person_name):
    """Extract frames from video and save detected faces using improved face detection"""
    # Configure extraction parameters to match previous behavior while using improved detection
    output_folder = "reference_faces"  # Keep the same output folder structure
    save_limit = 15  # Keep same number of faces as before
    skip_frames = 4  # Reasonable frame skip value
    similarity_threshold = 0.98  # High threshold to ensure diverse faces

    try:
        print(f"\nExtracting faces for {person_name}...")
        extract_faces(
            name=person_name,
            video_path=video_path,
            output_folder=output_folder,
            save_limit=save_limit,
            skip_frames=skip_frames,
            similarity_threshold=similarity_threshold
        )
        
        # Verify we got some faces
        person_folder = Path(output_folder) / person_name
        face_count = len(list(person_folder.glob("*.jpeg")))
        print(f"\nExtracted {face_count} faces for {person_name}")
        return face_count > 0
        
    except Exception as e:
        print(f"Error extracting faces for {person_name}: {str(e)}")
        return False

def update_detection_script(new_model_path):
    """Update the model path in detection_siamese.py"""
    detection_path = Path("siamese/detection_siamese.py")
    if not detection_path.exists():
        print("Warning: Could not find detection_siamese.py")
        return False
        
    with open(detection_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'siamese_net.load_state_dict(torch.load(' in line:
            lines[i] = f'siamese_net.load_state_dict(torch.load(r"{new_model_path}", map_location=device))\n'
            break
    
    with open(detection_path, 'w') as f:
        f.writelines(lines)
    
    print(f"Updated detection script to use {new_model_path}")
    return True

def collect_other_people_folders():
    """Collect existing reference face folders to use as negative examples"""
    reference_folder = Path("reference_faces")
    if not reference_folder.exists():
        return []
    
    other_folders = []
    for folder in reference_folder.iterdir():
        if folder.is_dir() and len(list(folder.glob("*.jpeg"))) > 0:
            other_folders.append(str(folder))
    
    return other_folders

def process_videos(videos_dict, use_improved_training=True):
    """Process multiple videos and train model for new people"""
    if not videos_dict:
        print("No videos provided!")
        return
    
    # Extract frames for all videos first
    successful_extractions = []
    for person_name, video_path in videos_dict.items():
        if not os.path.exists(video_path):
            print(f"Error: Video file not found for {person_name}: {video_path}")
            continue
            
        if extract_frames(video_path, person_name):
            successful_extractions.append(person_name)
    
    if not successful_extractions:
        print("No faces were successfully extracted from any video!")
        return
    
    # Backup current model
    current_model = "best_siamese_model.pth"
    backup_model = "best_siamese_model_backup.pth"
    if os.path.exists(current_model):
        shutil.copy2(current_model, backup_model)
        print(f"\nBacked up current model to {backup_model}")
    
    # Collect existing people folders for negative examples
    all_other_folders = collect_other_people_folders()
    
    # Train for all new people
    try:
        for person_name in successful_extractions:
            print(f"\nStarting transfer learning for {person_name}...")
            
            # Get other people's folders (excluding current person)
            other_folders = [f for f in all_other_folders if not f.endswith(person_name)]
            
            if use_improved_training:
                # Use improved training script with proper arguments
                cmd = [
                    "python",
                    "people_registration/add_new_person.py",  # Updated script path
                    person_name,
                    "--model", current_model,
                    "--target-images", f"reference_faces/{person_name}",
                    "--epochs", "15",
                    "--batch-size", "8",
                    "--lr", "0.0001",
                    "--pairs-per-epoch", "1000"
                ]

                
                # Add other people's folders if available
                if other_folders:
                    cmd.extend(["--other-images"] + other_folders)
                    print(f"Using {len(other_folders)} other people as negative examples")
                else:
                    print("No other people found, using synthetic negatives")
                
                # Add validation if we have enough data
                if len(other_folders) >= 2:
                    cmd.append("--validate")
                
            else:
                # Fallback to original training script format
                cmd = [
                    "python",
                    "people_registration/add_new_person.py",  # Still using improved, fallback also from people_registration
                    person_name,
                    "--model", current_model,
                    "--images", f"reference_faces/{person_name}"
                ]

            
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
            # Update current model path for next person
            if use_improved_training:
                latest_model = f"best_{person_name.lower().replace(' ', '_')}_model.pth"
            else:
                latest_model = f"finetuned_{person_name.lower()}_model.pth"
                
            if os.path.exists(latest_model):
                shutil.copy2(latest_model, current_model)
                print(f"Updated {current_model} with {person_name}'s trained model")
                # Add current person to other folders for next iterations
                all_other_folders.append(f"reference_faces/{person_name}")
        
        # Update detection script with final model
        update_detection_script(current_model)
        print("\nProcess completed successfully!")
        print(f"Final model incorporates all {len(successful_extractions)} people")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during training: {str(e)}")
        if os.path.exists(backup_model):
            shutil.copy2(backup_model, current_model)
            print(f"Restored backup model from {backup_model}")
    except FileNotFoundError as e:
        print(f"\nTraining script not found: {str(e)}")
        print("Make sure the improved add_new_person.py is in the correct location")

def interactive_mode():
    """Interactive mode for user input"""
    videos = {}
    print("=== Batch Face Recognition Training ===")
    print("Enter person names and their video paths.")
    print("Press Enter without typing a name to finish.\n")
    
    while True:
        person = input("Enter person name (or press Enter to finish): ").strip()
        if not person:
            break
        video = input(f"Enter video path for {person}: ").strip().strip('"')
        if video:
            videos[person] = video
        else:
            print("No video path provided, skipping...")
    
    return videos

if __name__ == "__main__":
    # Check if improved training script exists
    improved_script_exists = os.path.exists("people_registration/add_new_person.py")
    original_script_exists = os.path.exists("people_registration/add_new_person.py")  # same path since both are in same dir now

    
    if not improved_script_exists and not original_script_exists:
        print("Error: No training script found!")
        print("Please ensure either 'add_new_person.py' (improved) or 'siamese/add_new_person.py' (original) exists")
        exit(1)
    
    use_improved = improved_script_exists
    print(f"Using {'improved' if use_improved else 'original'} training script")
    
    # Example usage - uncomment to use hardcoded videos
     
    videos = {
         "John": r"C:\Users\OWNER\Desktop\WhatsApp Video 2025-06-23 at 09.34.06_dc6cd510.mp4",
    #     "Emma": r"C:\Videos\emma_video.mp4",
    #     "Mike": r"C:\Videos\mike_video.mp4"
     }
    
    # Interactive mode
    #videos = interactive_mode()
    
    if videos:
        print(f"\nProcessing {len(videos)} videos...")
        for name, path in videos.items():
            print(f"  {name}: {path}")
        
        confirm = input("\nProceed with training? (y/n): ").strip().lower()
        if confirm in ['y', 'yes']:
            process_videos(videos, use_improved_training=use_improved)
        else:
            print("Training cancelled.")
    else:
        print("No videos to process.")