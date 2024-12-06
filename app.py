from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import json

# models
model = tf.keras.models.load_model('./models/conv_model.keras')
unlearned_model = tf.keras.models.load_model('./models/unlearned_model.keras')

app = Flask(__name__)
@app.route('/')
def index():
    """
    index route
    :return: index.html
    """
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    route to process predictions
    receives an image
    processes image, converts array into tensor, makes prediction
    :return: JSON; prediction of what number the input image was
    """
    # get data and process image
    data = request.get_json()
    image = data.get('image')
    processed = process_image(image)

    # convert new array to tensor and normalize it
    tensor = tf.convert_to_tensor(processed, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis=0)
    # tensor = tf.keras.utils.normalize(tensor)
    tensor = tensor / 255

    # make predictions with both models
    predictions = model.predict(tensor)
    unlearned_prediction = unlearned_model.predict(tensor)
    return {"prediction": json.dumps(np.argmax(predictions).astype(np.int32).tolist()),
            "certainty": (predictions[0][np.argmax(predictions)] * 100).tolist(),
            "unlearned_prediction": json.dumps(np.argmax(unlearned_prediction).astype(np.int32).tolist()),
            "unlearned_certainty": (unlearned_prediction[0][np.argmax(unlearned_prediction)] * 100).tolist(),
            }

def process_image(image_data):
    """
    Takes image and transforms it into a 28x28 numpy array with 0-255 values (black and white)
    :param image_data: image to be processed
    :return: 28x28 array w/ values 0-255
    """
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))
    img = img.convert("L")
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img)
    return img_array.tolist()


if __name__ == '__main__':
    app.run()
