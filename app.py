from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from utils import prepare_image, model, decode_predictions

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file selected")

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction="No file selected")

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = prepare_image(filepath)
            preds = model.predict(img)
            result = decode_predictions(preds, top=1)[0][0]
            prediction = f"{result[1].capitalize()} ({round(result[2]*100, 2)}%)"
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
