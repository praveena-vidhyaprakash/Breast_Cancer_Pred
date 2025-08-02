import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('breast_cancer.csv')

X = df.drop('target', axis=1)
y = df['target']

model = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
model.fit(X, y)

print("Saving the model to 'model.pkl'...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved successfully.")