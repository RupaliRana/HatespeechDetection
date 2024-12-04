import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
# nltk.download('stopwords')
from nltk.util import pr
stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))

df=pd.read_csv("twitter_data.csv")
# print(df.head())

df['labels']=df['class'].map({0:"Hate Speech Detected",1:"Offensive language detected",2:"No hate and offensive speech"})
# print(df.head())

df=df[['tweet','labels']]
df.head()

def clean(text):
    text=str(text).lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ')if word not in stopword]
    text=" ".join(text)
    text=[stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
df["tweet"]=df["tweet"].apply(clean)
# print(df.head())

x=np.array(df["tweet"])
y=np.array(df["labels"])
cv=CountVectorizer()
x=cv.fit_transform(x)
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)
clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)

test_data="skinny"
df=cv.transform([test_data]).toarray()
print(clf.predict(df))

# pickell file
pickle.dump(clf,open('classifier_file.pkl','wb'))
pickle.dump(cv,open('cv.pkl','wb'))