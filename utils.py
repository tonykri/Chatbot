from nltk.corpus import stopwords

stopwords = stopwords.words('english')

def removeStopwords(sentence):
    querywords = sentence.split()
    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    return ' '.join(resultwords)