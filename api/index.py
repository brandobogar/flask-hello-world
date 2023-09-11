import os
from flask import Flask, request, abort, jsonify 
from werkzeug.utils import secure_filename
import tensorflow as tf
from flask_cors import CORS

from prediction import prediction
from create_dataset import create_datates
from get_latest import get_latest_uploads, get_latest_result

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif']
app.config['UPLOAD_PATH'] = 'dataset/uploads'


@app.route("/")
def index():
    return jsonify("Hello world")


@app.route('/classification', methods=['POST', 'GET'])
def classification():
    if request.method == 'POST':
        print('request files:', request.files)
        base_filename = 'image'
        extension = '.jpg'
        counter = 1
        newName = f'{base_filename}_{counter}{extension}'
        filePath = os.path.join(app.config['UPLOAD_PATH'], newName)

        uploaded_file = request.files['image']
        filename = secure_filename(uploaded_file.filename)
        print('filename:', filename)
        if filename != '':
            file_ext = os.path.splitext(filename)[1]
            if file_ext not in app.config["UPLOAD_EXTENSIONS"]:
                abort(400)

        while os.path.exists(filePath):
            counter += 1
            newName = f'{base_filename}_{counter}{extension}'
            filePath = os.path.join(app.config['UPLOAD_PATH'], newName)

        uploaded_file.save(filePath)
        create_datates()
        print("berhasil")

        response_data = {
            "message": 'image processed succesfully',
            'citra_path': filePath
        }
        return jsonify(response_data), 200

    elif request.method == 'GET':
        citra = get_latest_uploads()
        data_tes = tf.keras.preprocessing.image_dataset_from_directory(
            'dataset/data_tes/',
            image_size=(224, 224),
            shuffle=False,
            batch_size=32)

        result = get_latest_result()
        print(result)

        total_pala, pala_a, pala_b, pala_c, = prediction(
            citra_tes=citra, data_tes=data_tes)

        data = {
            'total_pala': str(total_pala),
            'pala_a': str(pala_a),
            'pala_b': str(pala_b),
            'pala_c': str(pala_c),
            result: str(result)
        }

        return jsonify(data)

        


# if __name__ == '__main__':
#     app.run(debug=True)
