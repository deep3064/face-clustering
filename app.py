import os
import shutil
import face_recognition
import cv2
import time
import pickle  # To save and load face data
from flask import Flask, render_template, request

app = Flask(__name__)

# Configuration
INPUT_DATABASE = "PIC"
RESULTS_FOLDER = os.path.join('static', 'results')
UPLOAD_FOLDER = os.path.join('static', 'uploads')
CACHE_FILE = "face_cache.pkl" # Where we store "known" faces
TOLERANCE = 0.55 
SCALE_FACTOR = 0.5 

for folder in [INPUT_DATABASE, RESULTS_FOLDER, UPLOAD_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def get_face_library():
    """Loads existing encodings or scans new photos if they aren't cached."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            library = pickle.load(f)
    else:
        library = {}

    updated = False
    current_files = [f for f in os.listdir(INPUT_DATABASE) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"[*] Checking for new photos in {INPUT_DATABASE}...")
    
    for filename in current_files:
        # If photo is NOT in our library, scan it now
        if filename not in library:
            print(f"  [+] New photo detected: {filename}. Scanning...")
            path = os.path.join(INPUT_DATABASE, filename)
            
            img = cv2.imread(path)
            if img is None: continue
            
            small_img = cv2.resize(img, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
            rgb_small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_img, model="hog")
            encodings = face_recognition.face_encodings(rgb_small_img, face_locations)
            
            # Store even if 0 faces found (so we don't re-scan empty photos)
            library[filename] = encodings
            updated = True

    # Save library back to disk if we added new photos
    if updated:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(library, f)
            print("[*] Face library updated and saved.")

    return library

@app.route('/', methods=['GET', 'POST'])
def index():
    images, matches, time_taken = [], 0, 0
    search_performed = False
    error_message = None

    if request.method == 'POST':
        start_time = time.time()
        search_performed = True
        
        try:
            file = request.files['sample_photo']
            sample_path = os.path.join(UPLOAD_FOLDER, "sample.jpg")
            file.save(sample_path)

            sample_img = face_recognition.load_image_file(sample_path)
            sample_encs = face_recognition.face_encodings(sample_img)

            if not sample_encs:
                error_message = "No face detected in sample."
            else:
                target_enc = sample_encs[0]
                
                # 1. Get Library (Only scans NEW photos)
                library = get_face_library()

                # 2. Clear old results
                for f in os.listdir(RESULTS_FOLDER):
                    os.unlink(os.path.join(RESULTS_FOLDER, f))

                # 3. Compare with Cache (Super Fast)
                for filename, encs in library.items():
                    for enc in encs:
                        match = face_recognition.compare_faces([target_enc], enc, tolerance=TOLERANCE)
                        if match[0]:
                            shutil.copy2(os.path.join(INPUT_DATABASE, filename), os.path.join(RESULTS_FOLDER, filename))
                            matches += 1
                            break
                
                images = os.listdir(RESULTS_FOLDER)
                time_taken = round(time.time() - start_time, 2)

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template('index.html', images=images, matches=matches, 
                           search_performed=search_performed, error_message=error_message, 
                           time_taken=time_taken)

if __name__ == '__main__':
    app.run(debug=True)