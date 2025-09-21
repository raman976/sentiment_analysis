# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# import joblib

# results={}
# def train_classical_models(X_train,Y_train,X_test,Y_test):
#     import os
#     if not os.path.exists("../models"):
#         os.makedirs("../models")


#     ##Vectorizer##
#     vectorizer=TfidfVectorizer(max_features=5000)
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)

#     ##---------##

#     ##Logistic_Regression##
#     logistic_model=LogisticRegression()
#     logistic_model.fit(X_train_vec,Y_train)
#     results['logistic_class']=logistic_model.score(X_test_vec,Y_test)
#     ##------------------##

#     ##NaiveByes##
#     NB_model=MultinomialNB()
#     NB_model.fit(X_train_vec,Y_train)
#     results['naive_bayes']=NB_model.score(X_test_vec,Y_test)





#     joblib.dump(logistic_model, "../models/logistic_class.pkl")
#     joblib.dump(NB_model, "../models/naive_bayes.pkl")
#     joblib.dump(vectorizer, "../models/tfidf.pkl")

#     return results
    
    
# if __name__=="__main__":
#     results.update(train_classical_models(X_train,Y_train,X_test,Y_test))
#     print('final comparison',results)













def train_and_save_models(X_train, Y_train, X_test, Y_test):
    results = {}

    if not os.path.exists("../models"):
        os.makedirs("../models")

    # ----- Classical Models -----
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_vec, Y_train)
    results['logistic_class'] = logistic_model.score(X_test_vec, Y_test)

    # Naive Bayes
    NB_model = MultinomialNB()
    NB_model.fit(X_train_vec, Y_train)
    results['naive_bayes'] = NB_model.score(X_test_vec, Y_test)

    # Save classical models
    joblib.dump(logistic_model, "../models/logistic_class.pkl")
    joblib.dump(NB_model, "../models/naive_bayes.pkl")
    joblib.dump(vectorizer, "../models/tfidf.pkl")

    # ----- BERT Model -----
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(set(Y_train))
    )

    train_dataset = SentimentDataset(X_train, Y_train, tokenizer)
    val_dataset = SentimentDataset(X_test, Y_test, tokenizer)

    # Compatible with transformers 4.56.2
    training_args = TrainingArguments(
    output_dir="../models/bert_model",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    report_to=[]  # <-- disables wandb / all logging
  )


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    tokenizer.save_pretrained("../models/bert_model")
    model.save_pretrained("../models/bert_model")

    # Evaluate BERT
    predictions = trainer.predict(val_dataset)
    bert_preds = predictions.predictions.argmax(-1)
    results['bert'] = (bert_preds == Y_test).mean()

    return results, bert_preds


