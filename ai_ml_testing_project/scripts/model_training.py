import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model(data):
    """Train a Random Forest model."""
    X = data[['age', 'income']]
    y = data['class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.2f}")
    return model, X_test, y_test

if __name__ == "__main__":
    data = pd.read_csv("../data/sample_data.csv")
    model, X_test, y_test = train_model(data)
