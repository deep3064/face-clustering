import face_recognition
import os
import shutil
import numpy as np
from sklearn.cluster import DBSCAN

# --- CONFIGURATION ---
INPUT_DIR = "input_photos"
OUTPUT_DIR = "output_clusters"
# eps: 0.4 (strict) to 0.6 (loose). 0.5 is usually perfect.
EPS_VALUE = 0.5 
MIN_SAMPLES = 2

def organize_faces():
    # --- STEP 0: AUTO-DELETE OLD RESULTS ---
    if os.path.exists(OUTPUT_DIR):
        print(f"[*] Cleaning up old results in '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(OUTPUT_DIR)

    # --- STEP 1: ENCODING ---
    encodings = []
    image_paths = []

    if not os.path.exists(INPUT_DIR):
        print(f"[!] Error: The folder '{INPUT_DIR}' does not exist.")
        return

    print(f"[*] Scanning {INPUT_DIR} and encoding faces...")
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(INPUT_DIR, filename)
            try:
                image = face_recognition.load_image_file(path)
                # Find face encodings
                face_encs = face_recognition.face_encodings(image)
                
                if len(face_encs) > 0:
                    # We process the first face found in each photo
                    encodings.append(face_encs[0])
                    image_paths.append(path)
                    print(f"  [+] Encoded: {filename}")
                else:
                    print(f"  [-] No face found: {filename}")
            except Exception as e:
                print(f"  [!] Error processing {filename}: {e}")

    if not encodings:
        print("[!] No faces were found. Please add photos to 'input_photos'.")
        return

    # --- STEP 2: CLUSTERING ---
    print(f"\n[*] Clustering {len(encodings)} images using DBSCAN...")
    cluster_model = DBSCAN(eps=EPS_VALUE, min_samples=MIN_SAMPLES, metric="euclidean")
    cluster_model.fit(encodings)

    # --- STEP 3: ORGANIZATION ---
    labels = cluster_model.labels_
    unique_labels = np.unique(labels)
    
    print(f"[*] Organizing files into folders...")
    for label_id in unique_labels:
        # Define folder name
        if label_id == -1:
            folder_name = "Unknown_or_Single_Photos"
        else:
            folder_name = f"Person_{label_id}"
            
        current_dir = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(current_dir, exist_ok=True)

        # Move files belonging to this cluster
        indices = np.where(labels == label_id)[0]
        for i in indices:
            src_path = image_paths[i]
            shutil.copy2(src_path, current_dir)

    # --- FINAL SUMMARY ---
    print("\n" + "="*30)
    print("      PROJECT SUMMARY")
    print("="*30)
    print(f"Total Photos Processed: {len(image_paths)}")
    print(f"Unique People Identified: {len([l for l in unique_labels if l != -1])}")
    print(f"Results Saved To: {OUTPUT_DIR}")
    print("="*30)

if __name__ == "__main__":
    organize_faces()