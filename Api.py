from flask import Flask 
from flask import render_template
from flask import request
from flask import jsonify
from flask import send_from_directory
from PIL import Image
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

seg_model = YOLO('Segmentation_Model.pt')
dectect_model = YOLO('Detection_Model.pt')

@app.route('/')
def index():
    return render_template('Main.html')

@app.route('/segmentation_page')
def segmentation_page():
    return render_template('Segmentation.html')

@app.route('/detection_page')
def detection_page():
    return render_template('Detection.html')

@app.route('/about')
def about():
    return render_template('About.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        img_file = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'Uploads', img_file.filename)
        img_file.save(upload_path)

        img = cv2.imread(upload_path)
        results = dectect_model.predict(img, save=True)

        # Find the latest output directory
        output_dir = os.path.join(basepath, 'runs', 'detect')
        latest_dir = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
        output_image_path = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.endswith('.jpg')][0]

        # Return the relative path for the static file handler
        relative_output_path = os.path.relpath(output_image_path, basepath)

        return jsonify({'outputImagePath': relative_output_path})
    
@app.route('/segmentation', methods=['POST'])
def segment():
    if 'file' in request.files:
        img_file = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'Uploads', img_file.filename)
        img_file.save(upload_path)

        img = cv2.imread(upload_path)
        results = seg_model.predict(img, save=True)

        output_dir = os.path.join(basepath, 'runs', 'segment')
        latest_dir = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))], key=os.path.getmtime)
        output_image_path = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if f.endswith('.jpg')][0]

        relative_output_path = os.path.relpath(output_image_path, basepath)

        return jsonify({'outputImagePath': relative_output_path})

@app.route('/<path:filename>')
def serve_file(filename):
    return send_from_directory('.', filename)


if __name__ == '__main__':
    app.run(debug=True)



