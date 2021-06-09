from flask import Flask, request
from flask_restful import Resource, Api


class Sbert(Resource):

    def get(self):
        produkt1 = request.args.get('produkt1')
        produkt2 = request.args.get('produkt2')
        return {"produkt1": produkt1, "produkt2": produkt2}
