from flask import Flask, json, make_response, request
from flask_restful import Resource, Api
from sentence_transformers import SentenceTransformer, util


class Sbert(Resource):

    def get(self):
        produkt1 = request.args.get('produkt1')
        produkt2 = request.args.get('produkt2')

        matching_score = self.calculate_match([produkt1, produkt2])
        return {"Matching score": matching_score}

    def calculate_match(self, sentences):
        # https://www.sbert.net/
        # model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        model = SentenceTransformer('data/sbert_trained_model')

        # Compute embeddings
        embeddings = model.encode(sentences, convert_to_tensor=True)

        # Compute cosine-similarities for each sentence with each other sentence
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

        # Find the pairs with the highest cosine similarity scores
        pairs = []
        for i in range(len(cosine_scores) - 1):
            for j in range(i+1, len(cosine_scores)):
                pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

        # Sort scores in decreasing order
        pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
        matching_score = {}

        for pair in pairs[0:10]:
            i, j = pair['index']
            matching_score = "Matching score: {:.4f}".format(pair['score'])

        return matching_score
