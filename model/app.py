from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
import os
import fitz,sys

# load the model from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		file = request.form['myfile']
		fname = file
		doc = fitz.open(fname)
		text = ""
		for page in doc:
			text = text + str(page.getText())
		tx = " ".join(text.split('\n'))
		doc = clf(tx)
		prediction = []
		for ent in doc.ents:
			prediction.append((f'{ent.label_.upper():{30}}- {ent.text}'))

	return render_template('result.html',prediction = prediction)


if __name__ == '__main__':
	app.run(debug=True)




