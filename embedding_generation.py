import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

data = pd.read_csv('cleaned_training.csv')
sentences = data['content']
sentences = sentences.to_numpy().tolist()

embeds = []

for i in range(len(sentences)):
    if i%500 == 0:
        print(f"{i}/{len(sentences)}")
    embeds.append(model.encode(str(sentences[i])))

data['embeds'] = embeds

data.to_csv('data_with_embeds.csv')