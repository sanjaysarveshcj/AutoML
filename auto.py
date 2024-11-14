import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

titanic = h2o.upload_file('Titanic-Dataset.csv')

print(titanic.head())

titanic['Survived'] = titanic['Survived'].asfactor()
x = titanic.columns
y = 'Survived'
x.remove(y)

train, test = titanic.split_frame(ratios=[0.8], seed=42)

aml = H2OAutoML(max_models=20, seed=42, max_runtime_secs=600)
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard
print('Best Model: ',lb)

preds = aml.predict(test)
print(preds.head())

perf = aml.leader.model_performance(test)
print(perf)

h2o.shutdown(prompt=False)
