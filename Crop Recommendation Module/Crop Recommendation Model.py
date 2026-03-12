# Crop Recommendation Model

# Install Libraries
!pip install pandas numpy scikit-learn joblib

# Import Libraries
import pandas as pd
import numpy as np
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Upload Dataset
uploaded = files.upload()

# Load Dataset
df = pd.read_csv("Crop Recommendation.csv")
df.head()

# Data Exploration
df.info()
df.describe()
df['label'].value_counts()

# Feature Engineering
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Save Model
joblib.dump(model, "crop_recommendation_model.pkl")
joblib.dump(scaler, "crop_scaler.pkl")
joblib.dump(label_encoder, "crop_label_encoder.pkl")

# Download models
files.download("crop_recommendation_model.pkl")
files.download("crop_scaler.pkl")
files.download("crop_label_encoder.pkl")
