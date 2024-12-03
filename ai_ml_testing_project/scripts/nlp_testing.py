import os
from checklist.perturb import Perturb
from checklist.test_suite import TestSuite
from transformers import pipeline

def nlp_test_suite():
    """Test sentiment analysis model with Checklist."""
    model = pipeline("sentiment-analysis")
    
    suite = TestSuite()
    data = ["The product is great!", "I hate this so much!", "It's okay, not bad."]
    
    # Add tests
    suite.add_test("Positive Sentiment", Perturb.change_names(data))
    suite.add_test("Negative Sentiment", Perturb.add_typos(data))
    
    # Evaluate
    results = suite.run(model)
    
    # Ensure the results directory exists
    os.makedirs("../results", exist_ok=True)
    
    with open("../results/checklist_report.txt", "w") as f:
        f.write(str(results))
    print("Checklist report saved!")

if __name__ == "__main__":
    nlp_test_suite()
