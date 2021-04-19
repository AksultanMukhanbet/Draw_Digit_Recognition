from flask import Flask
from flask import request
import keras

import image_ops

app = Flask(__name__)
# load model from file
model = keras.models.load_model("model.h5")


@app.route('/')
def hello():
    """
    Serves a page to draw digits on and classify them
    """

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

    # convert post data to PIL Image
    canvas_img = image_ops.post_data_to_image(image_data, img_size)
    # crop, resize, and translate the drawn digit to match digits from the MNIST data set as closely as possible
    mnistified_img = image_ops.mnistify_image(canvas_img, save_stages=False)
    # convert the image to a form that the model accepts as input
    model_input = image_ops.image_to_model_input(mnistified_img)

    # predict and return the digit classification
    #print(f"model_input: {model_input}")
    model_output = model.predict_classes(model_input)
    print(f"model_output: {model_output[0]}")
    preds = model.predict_proba(model_input, batch_size=32, verbose=1)
    print(f"preds: {preds[0]}")
    return str(model_output[0])


if __name__ == '__main__':
    app.run(host="0.0.0.0", threaded=False)
