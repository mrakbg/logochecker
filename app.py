from flask import Flask, request, render_template
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained MobileNet model
model = MobileNet(weights='imagenet')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)

            # Preprocess the image for prediction
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Run the model and get predictions
            predictions = model.predict(img_array)
            results = decode_predictions(predictions, top=3)[0]

            # Display results
            response = f"<h3>Top predictions:</h3><ul>"
            for pred in results:
                response += f"<li>{pred[1]} ({pred[2]*100:.2f}%)</li>"
            response += "</ul>"
            return response
    return render_template('upload.html')

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
