import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load a sample dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Train a simple model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Check accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy:.2f}")

# 4. Save the model as model.pkl
joblib.dump(model, "model.pkl")
print("model.pkl saved successfully!")