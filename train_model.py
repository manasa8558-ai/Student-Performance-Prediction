
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv("student-mat.csv", sep=";")

# Step 2: Select input columns
features = [
    "G1", "G2", "studytime",
    "failures", "absences"
]

X = df[features]
y = df["G3"]

# Step 3: Divide data for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the ML model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Test the model
predictions = model.predict(X_test)

print("Training completed!")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 2))

# Step 6: Save the trained model
joblib.dump(model, "student_model.pkl")

print("Model saved successfully!")
