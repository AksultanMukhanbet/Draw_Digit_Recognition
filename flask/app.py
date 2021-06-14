from flask import Flask
from flask import request
import keras

import image_ops

app = Flask(__name__)
model = keras.models.load_model("model.h5")


@app.route('/')
def hello():
    return app.send_static_file("digit-classifier.html")


@app.route("/classify", methods=["POST"])
def classify():
    """
    Classifies a drawn digit based on pixel data taken from the canvas
    :return: Digit classification (0-9)
    """

    img_size = int(request.values.get("size"))
    image_data = request.values.get("pixelData")
    print("image_data", type(image_data))

    canvas_img = image_ops.post_data_to_image(image_data, img_size)
    mnistified_img = image_ops.mnistify_image(canvas_img, save_stages=False)
    model_input = image_ops.image_to_model_input(mnistified_img)

    #print(f"model_input: {model_input}")
    model_output = model.predict_classes(model_input)
    print(f"model_output: {model_output[0]}")
    preds = model.predict_proba(model_input, batch_size=32, verbose=1)
    print(f"preds: {preds[0]}")
    return str(model_output[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=False)
