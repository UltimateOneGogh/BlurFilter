from flask import Flask, request, send_file
import cv2
import numpy as np
from utils import *

app = Flask(__name__)

checkpoint.restore(tf.train.latest_checkpoint("checkpoints/term_project/training_checkpoints/"))
model_blurr = checkpoint.generator


def get_image(data):
    jpg_as_np = np.frombuffer(data.read(), dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img


@app.route("/blur", methods=["GET", "POST"])
def blurr():
    src = get_image(request.files["src"])
    inp = image_to_tensor(src)
    preds = model_blurr(inp, training=False)
    return "ok"


@app.route('/style', methods=["GET", "POST"])
def style():
    style = get_image(request.files["style"])
    src = get_image(request.files["src"])


    return ""


app.run('0.0.0.0', port=5000, debug=True)
