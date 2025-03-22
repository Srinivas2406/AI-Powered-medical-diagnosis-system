import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load datasets
diabetes_data = pd.read_csv("database/diabetes.csv")
liver_data = pd.read_csv("database/liver.csv")
thyroid_data = pd.read_csv("database/thyroid.csv")

X_diabetes = diabetes_data.drop(columns=["Outcome"])
y_diabetes = diabetes_data["Outcome"]

imputer = SimpleImputer(strategy="mean")
X_diabetes.iloc[:, :] = imputer.fit_transform(X_diabetes)

X_train, X_test, y_train, y_test = train_test_split(X_diabetes, y_diabetes, test_size=0.2, random_state=42)

# Train model
diabetes_model = RandomForestClassifier(n_estimators=100, random_state=42)
diabetes_model.fit(X_train, y_train)

# Save model
joblib.dump(diabetes_model, "models/diabetes_model.pkl")

# Convert Gender column to numerical values
label_encoder = LabelEncoder()
liver_data["Gender"] = label_encoder.fit_transform(liver_data["Gender"])  # Male → 1, Female → 0

X_liver = liver_data.drop(columns=["Outcome"])
y_liver = liver_data["Outcome"]

X_liver.iloc[:, :] = imputer.fit_transform(X_liver)


X_train, X_test, y_train, y_test = train_test_split(X_liver, y_liver, test_size=0.2, random_state=42)

# Train model
liver_model = RandomForestClassifier(n_estimators=100, random_state=42)
liver_model.fit(X_train, y_train)

# Save model
joblib.dump(liver_model, "models/liver_model.pkl")

X_thyroid = thyroid_data.drop(columns=["Outcome"])
y_thyroid = thyroid_data["Outcome"]

X_thyroid.iloc[:, :] = imputer.fit_transform(X_thyroid)


X_train, X_test, y_train, y_test = train_test_split(X_thyroid, y_thyroid, test_size=0.2, random_state=42)

# Train model
thyroid_model = RandomForestClassifier(n_estimators=100, random_state=42)
thyroid_model.fit(X_train, y_train)

# Save model
joblib.dump(thyroid_model, "models/thyroid_model.pkl")

print("All models trained and saved successfully!")
