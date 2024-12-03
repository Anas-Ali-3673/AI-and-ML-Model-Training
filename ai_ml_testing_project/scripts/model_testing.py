import os
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def test_model(model, X_test, y_test):
    """Run DeepChecks suite on the model."""
    dataset = Dataset(X_test, label=y_test)
    suite = model_evaluation()
    result = suite.run(model, dataset)
    
    # Ensure the results directory exists
    os.makedirs("../results", exist_ok=True)
    
    result.save_as_html("../results/deepchecks_report.html")
    print("DeepChecks report saved!")

if __name__ == "__main__":
    from model_training import train_model
    
    data = pd.read_csv("../data/sample_data.csv")
    model, X_test, y_test = train_model(data)
    test_model(model, X_test, y_test)
