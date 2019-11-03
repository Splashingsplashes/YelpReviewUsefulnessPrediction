import pymongo
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["Yelp"]
ReviewsCol = mydb["ReviewData"]

ReviewDF = pd.DataFrame(list(ReviewsCol.find().limit(30000)))

score = [ str(int(row[5])) for row in ReviewDF.itertuples()]

ReviewDF["score"]=score


df = ReviewDF[['text','score']]
df = df[pd.notnull(df['text'])]

"""Exploration_score"""

# cnt_pro = df['score'].value_counts()
# plt.figure(figsize=(12,4))
# sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
# plt.ylabel('Number of Occurrences', fontsize=12)
# plt.xlabel('score', fontsize=12)
# plt.xticks(rotation=90)
# plt.show()


train, test = train_test_split(df, test_size=0.3, random_state=42)


"""Tokenize review content"""
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


train_tagged = train.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=r.score), axis=1)
test_tagged = test.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['text']), tags=r.score), axis=1)


print("1")

"""Building vocabulary"""

model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=4)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])


print("2")
"""Initialise model"""

for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

print("3")
"""Building the Final Vector Feature for the Classifier"""

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors



y_train, X_train = vec_for_learning(model_dbow, train_tagged)
y_test, X_test = vec_for_learning(model_dbow, test_tagged)


logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)


print("4")

y_pred = logreg.predict(X_test)

MSE_ErrorSum = 0
MAPE_Sum = 0
error = 0
for i in range(len(y_pred)):
    error = int(y_pred[i])-int(y_test[i])
    MSE_ErrorSum += abs(error)
    # MAPE_Sum += abs(error)/int(y_test[i])

MSE = MSE_ErrorSum / len(y_pred)
# MAPE = MAPE_Sum / len(y_pred)

print("MSE = " + str(MSE))
# print("MPAE = " + str(MAPE))
#
# from sklearn.metrics import accuracy_score, f1_score
# print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
# print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))