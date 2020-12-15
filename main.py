from flask import Flask, request, send_file
from skimage.util import img_as_ubyte
import cv2
import numpy as np
from utils import *
from style_models.utils import *
from style_models.style_model import *
from torchvision import models
import base64

app = Flask(__name__)

checkpoint.restore(tf.train.latest_checkpoint("/home/ruslan/ml_project_pm/BlurFilter/checkpoints/term_project/"
                                              "training_checkpoints/"))
model_blurr = checkpoint.generator
model_style = models.vgg19(pretrained=True).features.to(device).eval()


def get_image(data):
    jpg_as_np = np.frombuffer(data.read(), dtype=np.uint8)
    img = cv2.resize(cv2.imdecode(jpg_as_np, flags=1), (256, 256))
    return img


@app.route("/blur", methods=["GET", "POST"])
def blurr():
    src = get_image(request.files["src"])
    inp = image_to_tensor_(src)
    preds = model_blurr(inp, training=False)
    img = img_as_ubyte(np.array(preds[0] * 0.5 + 0.5))
    return {"result": base64.b64encode(img.tobytes())}


@app.route('/style', methods=["GET", "POST"])
def style():
    style = get_image(request.files["style"])
    src = get_image(request.files["src"])
    result = run_style_transfer(model_style, style, src)
    tensor = result.cpu().clone().squeeze(0)
    image = tensor_to_image(tensor)
    print(image)
    # return send_file(image.tobytes(), mimetype="image/gif")
    return {"result": base64.b64encode(image.tobytes())}


app.run('0.0.0.0', port=5000, debug=True)
