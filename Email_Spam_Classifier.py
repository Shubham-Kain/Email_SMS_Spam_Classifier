import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
import string
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score
import pickle

data = pd.read_csv("Project-9\spam.csv", encoding='latin-1')
data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
data.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
# print(data.sample(5))

le = LabelEncoder()
data['target'] = le.fit_transform(data['target'])
# print(data['target'].value_counts())
data.drop_duplicates(keep='first', inplace=True)
# print(data["target"].value_counts())
# plt.pie(data["target"].value_counts(), labels=["ham", "spam"], autopct='%0.2f%%')
# plt.legend(["ham", "spam"])
# plt.show()

# nltk.download('punkt_tab')
# nltk.download('stopwords')


data["num_character"]=data["text"].apply(len)
data["text"].apply(lambda x:len(nltk.word_tokenize(x)))
data["num_words"] = data["text"].apply(lambda x: len(nltk.word_tokenize(x)))
data["num_sentences"] = data["text"].apply(lambda x: len(nltk.sent_tokenize(x)))
# print(data[["num_character", "num_words", "num_sentences"]].describe())
# sns.histplot(data[data["target"]==0]['num_character'], color='blue', label='ham', )
# sns.histplot(data[data["target"]==1]['num_character'], color='red', label='spam')
# sns.pairplot(data,hue='target')
# plt.show()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    text = ' '.join(text)
    return text

data['transformed_text'] = data['text'].apply(transform_text)

# print(data.head(5))
spam_corpus = []
for msg in data[data['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
ham_corpus = []
for msg in data[data['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)

# print(Counter(spam_corpus).most_common(35))        
# print(Counter(ham_corpus).most_common(45))        

cv =CountVectorizer()
tfidf  = TfidfVectorizer(max_features=3000)
x = tfidf.fit_transform(data['transformed_text']).toarray()
y = data['target'].values
# print(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# gnb.fit(x_train, y_train)
# y_pred_gnb = gnb.predict(x_test)
# print("accuracy_score",accuracy_score(y_test, y_pred_gnb))
# print("confusion_matrix",confusion_matrix(y_test, y_pred_gnb))
# print("precision_score",precision_score(y_test, y_pred_gnb))

mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_test)
print("accuracy_score",accuracy_score(y_test, y_pred_mnb))
print("confusion_matrix",confusion_matrix(y_test, y_pred_mnb))
print("precision_score",precision_score(y_test, y_pred_mnb))

# bnb.fit(x_train, y_train)
# y_pred_bnb = bnb.predict(x_test)
# print("accuracy_score",accuracy_score(y_test, y_pred_bnb))
# print("confusion_matrix",confusion_matrix(y_test, y_pred_bnb))
# print("precision_score",precision_score(y_test, y_pred_bnb))

pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('spam_classifier.pkl', 'wb'))

# wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
# spam_wc=wc.generate(data[data['target'] == 1]['transformed_text'].str.cat(sep=" "))
# plt.figure(figsize=(12, 6))
# plt.imshow(spam_wc)