from flask import Flask, request, send_file


app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def hello_world():
    data = request.data
    print(data)
    return data


app.run('0.0.0.0', port=5000, debug=True)