from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from flask import Flask, request, jsonify
import sys
import os

app = Flask(__name__)

'''
Test with / make requests with 

Warning the data it was trained on was extremly unbalanced so it has issues predicting correctly
Will likely only predict. 


Note it is actually order sensative
curl -i -H "Content-Type: application/json" -X POST -d '{"age":50, "high_risk_exposure_occupation": false, "labored_respiration": true, "wheezes": true, "cough": false, "temperature": 35, "diarrhea": false,"fatigue": false, "headache": true, "loss_of_smell": false, "loss_of_taste": true, "runny_nose": false, "muscle_sore": true, "sore_throat": true}' http://localhost:5000
'''
@app.route('/', methods = ["POST"])
def host_model():
    data = request.json
    for key in data:
        data[key] = [data[key]]
        

    return jsonify({"prob": str(predict(pd.DataFrame.from_dict(data)))})

class customScaler:
    def __init__(self, X_train):
        X_train = X_train[["age", "temperature"]]
        self.scaler = preprocessing.StandardScaler().fit(X_train)

    def transform(self, data):
        transformed = self.scaler.transform(data[["age", "temperature"]])
        transformed = pd.DataFrame(data = transformed)
        data = data.drop(["age", "temperature"], axis=1)
        data = data.replace({False: 0, True: 1})
        data = pd.concat([data.reset_index(drop = True), transformed], axis = 1)
        return data

def predict(row):
    return model.predict_proba(scaler.transform(row))
    
#def main():    
data = pd.read_csv("data/data.csv")
data = data.drop(data.columns[0], axis = 1)
X, y = data.drop("covid19_test_results", axis=1), data[["covid19_test_results"]]
y = y.replace({"Negative": 0, "Positive": 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
    
scaler = customScaler(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state = 42).fit(X_train, y_train)

acc = model.score(X_test, y_test)
preds = model.predict(X_test)
cm = confusion_matrix(y_test, preds)
print("Model trained with acc of ", acc, file = sys.stderr)
print("Confusion matrix of ", file = sys.stderr)
print(cm,  file=sys.stderr)

app.run(host = "0.0.0.0", port = int(os.getenv(str("PORT"))))
    
    
#if __name__ == "__main__":
 #   main()






