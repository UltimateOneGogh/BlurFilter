from flask import Flask


app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    return "Как оно?"


app.run('0.0.0.0', port=5000, debug=True)