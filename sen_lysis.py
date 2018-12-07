from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap  
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import *
from sklearn.metrics import roc_auc_score
app = Flask(__name__)
Bootstrap(app)
@app.route('/')
def index():
	return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
	df = pd.read_csv("data/training.txt", sep='\t', names=['liked','txt'])
	stopset = set(stopwords.words('english'))
	vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents ='ascii',stop_words=stopset)
	y = df.liked
	X = vectorizer.fit_transform(df.txt)
        X_train, x_test,y_train,y_test = train_test_split(X, y, random_state=42)
        clf = naive_bayes.MultinomialNB()
        clf.fit(X_train, y_train)
        if request.method == 'POST':
                namequery = request.form['namequery']
    	        data = [namequery]
    	        our_input = np.array(data)
                vect = vectorizer.transform(our_input)
                our_prediction = clf.predict(vect)
        return render_template('result.html',prediction = our_prediction, name = namequery.upper())


if __name__ == '__main__':
	app.run(debug = True)






#training on reviews

#print(X)
#print(y.shape)
#print(X.shape)

#clf.fit(X_train, y_train)
#print(roc_auc_score(y_test, clf.predict_proba(x_test)[:,1]))
#our_input = np.array(["I would like to do more work in general"])
#vect = vectorizer.transform(our_input)
#print(clf.predict(vect))
