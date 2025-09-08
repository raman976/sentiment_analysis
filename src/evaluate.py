import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
def evaluate_model(model_name, X_test, y_test):
    if not os.path.exists("../results"):
        os.makedirs("../results")

    # Load models
    model = joblib.load(f"../models/{model_name}.pkl")
    vectorizer = joblib.load("../models/tfidf.pkl")

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    # Metrics
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig(f"../results/confusion_matrix_{model_name}.png")
    plt.close()

    # Save sample predictions
    sample_df = pd.DataFrame({
        "text": X_test,
        "actual": y_test,
        "predicted": y_pred
    })
    sample_df.to_csv(f"../results/sample_predictions_{model_name}.csv", index=False)
