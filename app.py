import os
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
api = Api(app)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ImageUpload(Resource):
    def post(self):
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # model = YOLO(os.path.join(os.path.dirname(__file__),'sst.pt'))
            model = YOLO(os.path.join(os.path.dirname(__file__),'bst.pt'))
            # model = YOLO('yolov8x.pt')
            otcm = model.predict(source=file_path, save=True)
            names = otcm[0].names
            classList = []
            coordinateList = []
            l = 0
            confCutOff = 0.20
            ml = 0
            isDetected = False
            isMulti = False
            for result in otcm:
                boxes = result.boxes  # Boxes object for bbox outputs
                masks = result.masks  # Masks object for segmenation masks outputs
                probs = result.probs 
            for i in range(len(boxes)):
                print('i is ', i)
                ar = boxes[i]
                if(ar.conf.item() > confCutOff):
                    isDetected = True
                    if(ml > 1):
                        isMulti = True
                    ml = ml + 1
                    classList.append({ "classStr":names[int(ar.cls)],"confidence":str(ar.conf.item())})
                    coordinateList.append({ "x1":str(ar.xyxy[0,0].item()),"y1":str(ar.xyxy[0,1].item()),"x2":str(ar.xyxy[0,2].item()),"y2":str(ar.xyxy[0,3].item())})
                    
            return jsonify({"success":"true","coordinateList":coordinateList,"classList":classList,"isMulti":isMulti, "isDetected":isDetected})

            # Optionally, you can process the image using the Pillow library.
            # For example, let's check the image size (width and height).
            with Image.open(file_path) as img:
                width, height = img.size

            return jsonify({"success": "File uploaded successfully.",
                            "filename": filename,
                            "image_size": f"{width}x{height}"})
        else:
            return jsonify({"error": "File type not allowed."}), 400

api.add_resource(ImageUpload, '/api/upload')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=5000, debug=False)