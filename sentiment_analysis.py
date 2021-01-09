import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import SGDClassifier

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

df = pd.read_csv('Amazon Review.csv')
df1 = df.dropna(subset=['Text'])
df1 = df1.dropna(subset=['Rating'])  
X = df1.iloc[:,1].values
y = df1.iloc[:,0].values
import numpy as np
Y=[0] * (len(y))
for i in range(0,(len(y))):
  if y[i]==1 or y[i]==2 or y[i]==3 or y[i]==4:
    Y[i] = 0              # Negative sentiments
  else:
    Y[i] = 1              # Positive sentiments
text_model = Pipeline([('tfidf',TfidfVectorizer(binary = True,max_df=0.611111111111111,norm = 'l2')),('model',SGDClassifier(loss="hinge",penalty="l2",tol=None,max_iter=5))])

text_model.fit(X,Y)

def u_in():
  text = st.text_input("Enter your review: ")
  a = []
  a.append(text)
  return a

st.title("Machine Learning Project")
st.title("Sentiment Analysis model ")
df = u_in()
pred = text_model.predict(df)  
if pred == 0:
  st.write("Negative Sentiment")
else:
  st.write("Positive Sentiment")
