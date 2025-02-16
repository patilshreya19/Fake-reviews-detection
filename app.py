

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import sklearn as sk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)
 

# Load your trained model
model=pickle.load(open('model.pkl','rb'))


@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
@app.route("/login")
def login():
	return render_template('login.html')

@app.route("/chart")
def chart():
	return render_template('chart.html')

@app.route("/performance")
def performance():
	return render_template('performance.html')

@app.route("/contact")
def contact():
	return render_template('contact.html')



@app.route("/index_FR",methods=['GET'])
def index():
	return render_template('index_FR.html')




@app.route('/predict', methods=['GET','POST'])
def predict():
    df = pd.read_csv('deceptive-opinion2.csv')
    df1 = df[['deceptive', 'text']]
    df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
    df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
    X = df1['text']
    Y = np.asarray(df1['deceptive'], dtype = int)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109)
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)
    message = request.form.get('enteredinfo')
    data = [message]
    vect = cv.transform(data).toarray()
    pred = model.predict(vect)

    return render_template('result_FR.html', prediction_text=pred)

 

if __name__ == '__main__':
    app.run(debug=True)

