#!flask/bin/python
from flask import Flask, jsonify
from flask import request
import pickle

app = Flask(__name__)
tasks = []
@app.route('/api1', methods=['GET'])
def get_tasks():
	return jsonify({'tasks': tasks})

@app.route('/api2', methods=['POST'])
def create_task():
	age = request.json['age']
	sex = request.json['sex']
	cp =  request.json['cp']
	trestbps =  request.json['trestbps']
	chol = request.json['chol']
	fbs = request.json['fbs']
	restecg =  request.json['restecg']
	thalach =  request.json['thalach']
	exang =  request.json['exang']
	oldpeak = request.json['oldpeak']
	slope =  request.json['slope']
	ca = request.json['ca']
	thal =  request.json['thal']
	#PROGRAM INFERENCE
	tasks.append([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak, slope, ca, thal])
	clf = pickle.load(open('model.sav', 'rb'))
	y_pred = clf.predict(tasks)
	#PROGRAM INFERENCE
	return jsonify({'prediksi':str(y_pred)}), 201

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5010')
