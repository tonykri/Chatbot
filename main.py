import numpy as np
import random
from data import data
from utils import removeStopwords
from MLmodel import MLmodelClass
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.utils import to_categorical 


# Collect all data and prepare to train
categories = []
xTrain = []
yTrain = []
temp = []
for category in data:
    categories.append(category["tag"])
    for sentence in category["patterns"]:
        result = removeStopwords(sentence)
        xTrain.append(result)
        temp.append(category["tag"])

for y in temp:
    yTrain.append(categories.index(y))


# Transform sentences
countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(xTrain)
tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

xTrainTfidf = xTrainTfidf.toarray()
yTrain = np.array(yTrain).reshape(-1,1)
yTrain = to_categorical(yTrain)


# Create and train the model
model = MLmodelClass(len(xTrainTfidf[0]), len(yTrain[0]))
model.train(xTrainTfidf, yTrain)


# Interaction with user
print("Hello how can I help you?")
while True:
    question = input()
    result = removeStopwords(question)
    xNewCounts = countVectorizer.transform([question])
    xNewTfidf = tfidTransformer.transform(xNewCounts)
    xNewTfidf = xNewTfidf.toarray()

    pred = model.predict(xNewTfidf)
    print(random.choice(data[pred]["responses"]))
    if (data[pred]["tag"] == "goodbye"): break
