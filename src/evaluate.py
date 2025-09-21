# import joblib
# from sklearn.metrics import classification_report, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# def evaluate_model(model_name, X_test, y_test):
#     if not os.path.exists("../results"):
#         os.makedirs("../results")

#     # Load models
#     model = joblib.load(f"../models/{model_name}.pkl")
#     vectorizer = joblib.load("../models/tfidf.pkl")

#     # Transform test data
#     X_test_vec = vectorizer.transform(X_test)
#     y_pred = model.predict(X_test_vec)

#     # Metrics
#     print(classification_report(y_test, y_pred))

#     # Confusion matrix
#     cm = confusion_matrix(y_test, y_pred)
#     sns.heatmap(cm, annot=True, fmt='d')
#     plt.savefig(f"../results/confusion_matrix_{model_name}.png")
#     plt.close()

#     # Save sample predictions
#     sample_df = pd.DataFrame({
#         "text": X_test,
#         "actual": y_test,
#         "predicted": y_pred
#     })
#     sample_df.to_csv(f"../results/sample_predictions_{model_name}.csv", index=False)











def evaluate_and_save_results(X_test, Y_test, bert_preds, le):
    if not os.path.exists("../results"):
        os.makedirs("../results")

    # Load classical models
    logistic_model = joblib.load("../models/logistic_class.pkl")
    NB_model = joblib.load("../models/naive_bayes.pkl")
    vectorizer = joblib.load("../models/tfidf.pkl")

    # Transform test data
    X_test_vec = vectorizer.transform(X_test)

    # Predict
    logistic_pred = logistic_model.predict(X_test_vec)
    NB_pred = NB_model.predict(X_test_vec)

    # Convert numeric predictions back to original labels
    logistic_pred_labels = le.inverse_transform(logistic_pred)
    NB_pred_labels = le.inverse_transform(NB_pred)
    bert_pred_labels = le.inverse_transform(bert_preds)
    Y_test_labels = le.inverse_transform(Y_test)

    for name, pred in zip(["logistic_class","naive_bayes","bert"], 
                          [logistic_pred_labels, NB_pred_labels, bert_pred_labels]):
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



