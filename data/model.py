from sentence_transformers import util,SentenceTransformer, models
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, InputExample, losses
import torch
from torch import nn
from torch.utils.data import DataLoader
#model = SentenceTransformer('distiluse-base-multilingual-cased')
#model = SentenceTransformer('distilbert-base-nli-mean-tokens')

import pandas as pd

df= pd.read_csv("./product_matching.csv")
sample=df.sample(n=50000, random_state=1)

#training sample
sample_200=df.sample(n=200, random_state=1)

#evaluation sample
sample_evaluation = df.sample(n=50, random_state=1)

#Define the model. Either from scratch of by loading a pre-trained model

# word_embedding_model = SentenceTransformer('data/sbert_trained_model/')

saved_model_path = 'data/sbert_trained_model'
word_embedding_model = models.Transformer('sentence-transformers/distilbert-base-nli-mean-tokens')

#pooling model 
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

print(pooling_model.get_sentence_embedding_dimension())
#dense model 
dense_model = models.Dense(pooling_model.get_sentence_embedding_dimension(), out_features=2,  activation_function=nn.Softmax())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
train_num_labels = len(sample_200)

import pathlib
current_abs_path = str(pathlib.Path().resolve())
saved_model_dir = current_abs_path + '/' + saved_model_path

import os.path
if os.path.isdir(saved_model_dir):
    print('model was already saved')
    model = SentenceTransformer(saved_model_dir)
else:
    print('model was not saved')


print(sample_200)
#transfer labels in float, otherwise it will not accepted
labels = []
for i in sample_200.match.values:
    labels.append(str(int(i)))   

#Define your train examples. You need more than just two examples...
train_examples = []
for sample_index, row in sample_200.iterrows():
    train_examples.append(InputExample(texts=[row.productname_1, row.productname_2], label=int(row.match)))

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.SoftmaxLoss(model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=train_num_labels)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=1,
          warmup_steps=100,
          evaluation_steps=13,
          output_path=saved_model_path)


#evaluate model first on trained data set
for sample_index, row in sample_200[0:10].iterrows():
    embeddings = model.encode([row.productname_1, row.productname_2])
    print(embeddings)
    print(row.match)
    print()
