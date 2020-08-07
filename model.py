import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

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
        
def main():
    data = pd.read_csv("data/data.csv")
    data = data.drop(data.columns[0], axis = 1)
    X, y = data.drop("covid19_test_results", axis=1), data[["covid19_test_results"]]
    y = y.replace({"Negative": 0, "Positive": 1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    scaler = customScaler(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    print(X_test)
    
if __name__ == "__main__":
    main()






