from sentence_transformers import util,SentenceTransformer, models
from sentence_transformers import evaluation
from sentence_transformers import SentenceTransformer, InputExample, losses
import math
from torch.utils.data import DataLoader
#model = SentenceTransformer('distiluse-base-multilingual-cased')
#model = SentenceTransformer('distilbert-base-nli-mean-tokens')

import pandas as pd

df= pd.read_csv("./product_matching.csv")

sample = df.sample(n=5000, random_state=1)
#70 percent of all data for training set
sample_training = sample.sample(frac=0.7, random_state=1)
#30 percent of all data for evaluation data set
sample_evaluation = sample.drop(sample_training.index)

#Define the model. Either from scratch of by loading a pre-trained model
# word_embedding_model = SentenceTransformer('data/sbert_trained_model/')
saved_model_path = 'data/sbert_trained_model_sim_score'
#bert model to create sentence representations
word_embedding_model = models.Transformer('sentence-transformers/distilbert-base-nli-mean-tokens')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

#sbert model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# retrieve the existing model or pull the defined if not exists
import pathlib
current_abs_path = str(pathlib.Path().resolve())
saved_model_dir = current_abs_path + '/' + saved_model_path

import os.path
if os.path.isdir(saved_model_dir):
    print('model was already saved')
    model = SentenceTransformer(saved_model_dir)
else:
    print('model was not saved')

#crete training and validation sets
train_num_labels = len(sample_training)

print(sample_training)


#transfer labels in float, otherwise it will not accepted
labels = []
for i in sample_training.match.values:
    labels.append(float(i))   

train_examples = []
for sample_index, row in sample_training.iterrows():
    train_examples.append(InputExample(texts=[row.productname_1, row.productname_2], label=float(row.match)))


labels_evaluation = []
for i in sample_evaluation.match.values:
    labels_evaluation.append(float(i))   

evaluation_examples = []
for sample_index, row in sample_evaluation.iterrows():
    evaluation_examples.append(InputExample(texts=[row.productname_1, row.productname_2], label=float(row.match)))

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sample_evaluation.productname_1.values, sample_evaluation.productname_2.values, sample_evaluation.match.values)

num_epochs = 4
#10% of train data for warm-up
warmup_steps = math.ceil(len(sample_training) * num_epochs * 0.1) 

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          evaluator=evaluator,
          evaluation_steps=500,         
          output_path=saved_model_path)


print("EVALUATION FIRST 10 training records")
#evaluate model first on trained data set
for sample_index, row in sample_training[0:10].iterrows():
    embeddings = model.encode([row.productname_1, row.productname_2])

    cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    print(cosine_scores)
    print(row.match)
    print()

print("EVALUATION FIRST 10 evalation records")
#evaluate model first on evaluation data set
for sample_index, row in sample_evaluation[0:10].iterrows():
    embeddings = model.encode([row.productname_1, row.productname_2])

    cosine_scores = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    print(cosine_scores)
    print(row.match)
    print()