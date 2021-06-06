import os
import json
import datetime
from flask import Flask
from flask import render_template, make_response, Response


app = Flask(__name__)

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_FOLDER, "resources")

@app.route('/')
@app.route('/index')
def start_app():
    return render_template('index.html')

@app.route('/about')
def about_action():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
