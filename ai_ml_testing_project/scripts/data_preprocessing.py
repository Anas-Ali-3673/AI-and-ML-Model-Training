import pandas as pd

def load_data(filepath):
    """Load and preprocess the dataset."""
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} rows of data.")
    return data

if __name__ == "__main__":
    data = load_data("../data/sample_data.csv")
    print(data.head())
