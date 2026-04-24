import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# load dataset
df = pd.read_csv('health_dataset_250.csv')

# encode sex
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# encode disease
disease_encoder = LabelEncoder()
df['Disease'] = disease_encoder.fit_transform(df['Disease'])

# features & target
X = df[['Age', 'Sex', 'Blood_Pressure', 'Heart_Rate', 'Body_Temperature', 'BMI']]
y = df['Disease']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# accuracy
print("Model Accuracy:", model.score(X_test, y_test))

# save model
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(disease_encoder, open('encoder.pkl', 'wb'))