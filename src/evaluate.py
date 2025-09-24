import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate_and_save_results(X_test, Y_test, distilbert_logits, le):
    """
    Evaluate classical models and DistilBERT predictions and save results.

    Args:
        X_test (list): Test texts
        Y_test (list or array): True labels (numeric)
        distilbert_logits (torch.Tensor or np.array): Model output logits
        le (LabelEncoder): Fitted label encoder
    """
    if not os.path.exists("../results"):
        os.makedirs("../results")

    # Load classical models
    logistic_model = joblib.load("../models/logistic_class.pkl")
    NB_model = joblib.load("../models/naive_bayes.pkl")
    vectorizer = joblib.load("../models/tfidf.pkl")

    # Transform test data for classical models
    X_test_vec = vectorizer.transform(X_test)

    # Predict using classical models
    logistic_pred = logistic_model.predict(X_test_vec)
    NB_pred = NB_model.predict(X_test_vec)

    # Predict using DistilBERT
    if isinstance(distilbert_logits, torch.Tensor):
        distilbert_pred = torch.argmax(distilbert_logits, dim=1).cpu().numpy()
    else:
        # Convert to numpy array if it isn’t already
        distilbert_pred = np.array(distilbert_logits)
        # If it’s 2D (logits), take argmax along axis=1
        if distilbert_pred.ndim > 1:
            distilbert_pred = distilbert_pred.argmax(axis=1)
        # If it’s already 1D (predictions), keep as is

    # Convert numeric predictions back to original labels
    logistic_pred_labels = le.inverse_transform(logistic_pred)
    NB_pred_labels = le.inverse_transform(NB_pred)
    distilbert_pred_labels = le.inverse_transform(distilbert_pred)
    Y_test_labels = le.inverse_transform(Y_test)

    # Evaluate and save results
    for name, pred in zip(["logistic_class","naive_bayes","distilbert"],
                          [logistic_pred_labels, NB_pred_labels, distilbert_pred_labels]):
        print(f"\nClassification Report for {name}:")
        print(classification_report(Y_test_labels, pred))
        cm = confusion_matrix(Y_test_labels, pred)
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title(f"Confusion Matrix: {name}")
        plt.show()

        # Save predictions
        sample_df = pd.DataFrame({
            "text": X_test,
            "actual": Y_test_labels,
            "predicted": pred
        })
        sample_df.to_csv(f"../results/sample_predictions_{name}.csv", index=False)
