from sentence_transformers import util,SentenceTransformer
model = SentenceTransformer('distiluse-base-multilingual-cased')
 
import pandas as pd


df= pd.read_csv("./product_matching.csv")
sample=df.sample(n=50000, random_state=1)
sample_200=df.sample(n=200, random_state=1)


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

print(sample_200)

labels = []
for i in sample_200.match.values:
    labels.append(float(i))

#Define your train examples. You need more than just two examples...
train_examples = [InputExample(texts=[sample_200.productname_1.values, sample_200.productname_2.values], label=labels)]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)