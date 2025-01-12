from flask import Flask, jsonify, Response, send_file
from flask_cors import CORS
import inference_classifier
import speech  # Import the speech module
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/start-camera', methods=['POST'])
def start_camera():
    # Call the start_camera function from inference_classifier
    inference_classifier.start_camera()
    return jsonify({'output': 'Camera Started'})

@app.route('/stop-camera', methods=['POST'])
def stop_camera():
    # Call the stop_camera function from inference_classifier
    inference_classifier.stop_camera()
    return jsonify({'output': 'Camera Stopped'})

@app.route('/start-detection', methods=['POST'])
def start_detection():
    # Call the start_detection function from inference_classifier
    inference_classifier.start_detection()
    return jsonify({'output': 'Detection Started'})

@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    # Call the stop_detection function from inference_classifier
    inference_classifier.stop_detection()
    return jsonify({'output': 'Detection Stopped'})

@app.route('/save-text', methods=['POST'])
def save_text():
    # Call the save_text function from inference_classifier
    inference_classifier.save_text()
    return jsonify({'output': 'Text Saved'})

@app.route('/convert-to-audio', methods=['POST'])
def convert_to_audio():
    # Call the text_to_speech function from speech module
    file_path = "detected_letters.txt"  # File containing the detected letters
    output_file = "webpage/output.mp3"  # Output MP3 file name
    speech.text_to_speech(file_path, output_file)
    return jsonify({'output': 'Text Converted to Audio'})

@app.route('/get-audio', methods=['GET'])
def get_audio():
    return send_file('webpage/output.mp3', mimetype='audio/mpeg')

@app.route('/clear-text', methods=['POST'])
def clear_text_route():
    inference_classifier.clear_text()
    return jsonify({'output': 'Text Cleared'})

@app.route('/add-space', methods=['POST'])
def add_space():
    inference_classifier.add_space()
    return jsonify({'output': 'Space Added'})

def generate_frames():
    while True:
        if inference_classifier.cap is None or not inference_classifier.cap.isOpened():
            continue
        success, frame = inference_classifier.cap.read()
        if not success:
            break

        # Process the frame for hand detection and prediction
        frame = inference_classifier.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8000)
