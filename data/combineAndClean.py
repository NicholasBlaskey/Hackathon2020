import pandas as pd
import os

# Read cols and combine them into single df
used_cols = ["covid19_test_results", "age", "high_risk_exposure_occupation",
            #"diabetes", "chd", "htn", "cancer", "asthma". "copd", "fever"
            #"autoimmune_dis", "smoker", "fever_calc", "rhonchi" "cough_severity", 
             "labored_respiration", "wheezes", "cough",
             "temperature", "diarrhea", "fatigue",
            "headache", "loss_of_smell", "loss_of_taste",
            "runny_nose", "muscle_sore", "sore_throat"]
path_to_data = "../../covidclinicaldata/data"
files =  os.listdir(path_to_data)
out = []
for i, f in enumerate(files):
    d = pd.read_csv(path_to_data + "/" + f)
    d = d[used_cols]
    assert(len(d.columns) == len(used_cols))
    if i == 0:
        out = d
    else:
        out = out.append(d)

print(out.covid19_test_results.value_counts())

# Remove nans
for c in ["labored_respiration", "wheezes"]:
    out[[c]] = out[[c]].fillna(value = False)
if "temperature" in out.columns:
    out[["temperature"]] = out[["temperature"]].fillna(
        value = out[["temperature"]].mean(skipna = True))    
out = out.dropna()

out.to_csv("data.csv")
