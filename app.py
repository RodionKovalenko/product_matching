import os
import json
import datetime
from flask import Flask
from flask import render_template, make_response, Response, request
from flask_restful import Resource, Api, reqparse
from src.webservice import sbert

app = Flask(__name__)
api = Api(app)
sbertAPI = sbert.Sbert()

BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
RESOURCE_DIR = os.path.join(BASE_FOLDER, "resources")


@app.route('/')
@app.route('/index')
def start_app():
    return render_template('index.html')


@app.route('/about')
def about_action():
    return render_template('about.html')

#when Button "Vergleich" is clicked
@app.route("/onComparestrings/", methods=['GET', 'POST'])
def on_compare_strings_action():
    produkt1 = request.args.get('produkt1')[:]
    produkt2 = request.args.get('produkt2')[:]

    product_match_result = sbertAPI.calculate_match([produkt1, produkt2])

    return render_template('index.html', product_matching_result=product_match_result, produkt1 = produkt1, produkt2 = produkt2)


# Sbert API
api.add_resource(sbertAPI, '/api/v1/comparestrings', endpoint='sbert')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
