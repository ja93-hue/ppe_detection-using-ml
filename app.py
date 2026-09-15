import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO # Import the YOLO class

# --- 1. Load the YOLOv8 Model and Set up for GPU Acceleration ---
def load_yolov8_model(model_path):
    """
    Loads a YOLOv8 model and moves it to the GPU if available.
    """
    # CORRECTED: Removed the extra '.cuda' from the function call.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load the YOLOv8 model using the dedicated YOLO class
        # It automatically detects the model type from the file extension (.pt)
        model = YOLO(model_path)
        # The YOLO class handles device placement and evaluation mode internally during predict().
        return model, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# --- 2. Initialize Flask App and Model ---
app = Flask(__name__)
# CORRECTED: Using a raw string (r"...") to handle the Windows file path correctly.
# Make sure to update this path to the correct location of your model.
MODEL_PATH = r"C:\Users\J\Downloads\final\8m.pt" 

# Load the model once when the application starts
model, device = load_yolov8_model(MODEL_PATH)
if model is None:
    print("FATAL: Model could not be loaded. The application will not run correctly.")
    
# --- 3. Webcam Capture Object (global) ---
# We'll initialize the camera once when the app starts
camera = None
try:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("WARNING: Could not open webcam. Is it in use by another application?")
except Exception as e:
    print(f"ERROR: Failed to initialize webcam: {e}")
    camera = None


# --- 4. Generator Function for Video Streaming ---
def gen_frames():
    """
    Generator function to continuously read frames, process them, and encode for streaming.
    """
    if model is None:
        # Yield an error message frame if the model failed to load
        error_msg = "Error: Model not loaded. Please check the model path and file."
        print(error_msg)
        # Create a simple black image with the error text
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, error_msg, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', err_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return # Stop the generator
    
    if camera is None or not camera.isOpened():
        # Yield a different error if the camera is not available
        error_msg = "Error: Webcam not available."
        print(error_msg)
        err_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(err_frame, error_msg, (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', err_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    while True:
        # Read a frame from the webcam
        success, frame = camera.read()
        if not success:
            print("Error: Failed to grab a frame from the camera.")
            # Break the loop if reading a frame fails
            break 
        else:
            # --- Mirror the Frame ---
            # Flips the image horizontally for a mirrored effect (like a mirror)
            mirrored_frame = cv2.flip(frame, 1)

            # --- Perform Inference on the Mirrored Frame ---
            # The YOLO class handles the inference with `predict()`.
            # `stream=True` returns a generator for continuous processing.
            # `device=device.type` ensures GPU acceleration if available.
            results = model.predict(
                source=mirrored_frame, 
                stream=True, 
                device=device.type, # Use 'cuda' or 'cpu' string
                verbose=False # Set to True for debugging logs
            )

            # --- Process and Display Results ---
            # Iterate through the results from the generator
            for r in results:
                # `r.plot()` is the recommended way to visualize results with YOLOv8.
                # It returns the annotated image as a numpy array.
                rendered_frame = r.plot()
                
                # Encode the frame as JPEG for streaming
                ret, buffer = cv2.imencode('.jpg', rendered_frame)
                frame_bytes = buffer.tobytes()

                # Yield the frame in a multipart/x-mixed-replace format
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 5. Flask Routes ---
@app.route('/')
def index():
    """
    Main route that renders the HTML template for the video stream.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Route that streams the video feed from the webcam with detection.
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 6. Run the Flask App ---
if __name__ == '__main__':
    # Make sure to release the camera when the app shuts down
    try:
        # Run the app. host='0.0.0.0' makes it accessible on the local network
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        # Release the camera when the application is terminated
        if camera is not None and camera.isOpened():
            print("Releasing camera.")
            camera.release()