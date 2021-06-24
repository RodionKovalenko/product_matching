from sentence_transformers import util,SentenceTransformer
from sentence_transformers import evaluation
model = SentenceTransformer('distiluse-base-multilingual-cased')
 
import pandas as pd


df= pd.read_csv("./product_matching.csv")
sample=df.sample(n=50000, random_state=1)

#training sample
sample_200=df.sample(n=200, random_state=1)

#evaluation sample
sample_evaluation = df.sample(n=50, random_state=1)


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

labels = []
for i in sample_200.match.values:
    labels.append(float(i))
    
    
# evaluation labels    
labels_evaluation = []
for i in sample_evaluation.match.values:
    labels_evaluation.append(float(i))

#Define your train examples. You need more than just two examples...
train_examples = []
for sample_index, row in sample_200.iterrows():
    train_examples.append(InputExample(texts=[row.productname_1, row.productname_2], label=float(row.match)))

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)


evaluator = evaluation.EmbeddingSimilarityEvaluator(sample_evaluation.productname_1.values, 
                                                    sample_evaluation.productname_2.values, 
                                                    labels_evaluation)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=1,
          warmup_steps=100,
          evaluator=evaluator, 
          evaluation_steps=1,
          output_path='sbert_trained_model')