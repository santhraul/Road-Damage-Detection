import io
import os
import json
from PIL import Image
import shutil

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#RESULT_FOLDER = os.path.join('static')
#app.config['RESULT_FOLDER'] = RESULT_FOLDER

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).autoshape()  # for PIL/cv2/np inputs and NMS
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last.pt')  # custom model

model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        print(file)
        if not file:
            return

        if '.mp4' in file.filename:
            print(os.path.abspath('detect.py'))
            file.save(os.path.join('uploaded_file', 'video_test.mp4'))
            if os.path.exists('static/video_test_out.mp4'):
                os.remove('static/video_test_out.mp4')
            os.system('python /home/ubuntu/cs2/yolov5/detect.py --source uploaded_file/video_test.mp4 --weights last.pt')
            os.system('ffmpeg -i runs/detect/exp/video_test.mp4 -vcodec h264 -acodec mp2 runs/detect/exp/video_test_out.mp4')
            shutil.copy('runs/detect/exp/video_test_out.mp4', 'static/video_test_out.mp4')
            shutil.rmtree('runs')
            return render_template('prediction_video.html')

        else:
            img_bytes = file.read()
            results = get_prediction(img_bytes)
            results.save()  # save as results1.jpg, results2.jpg... etc.
            shutil.copy('runs/hub/exp/image0.jpg', 'static/image0.jpg')
            shutil.rmtree('runs')
            return render_template('prediction.html')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8501)
