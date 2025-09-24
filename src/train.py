from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

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

    # ----- DistilBERT Model -----
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(set(Y_train))
    )

    train_dataset = SentimentDataset(X_train, Y_train, tokenizer, max_len=128)
    val_dataset = SentimentDataset(X_test, Y_test, tokenizer, max_len=128)

    training_args = TrainingArguments(
        output_dir="../models/distilbert_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,  # smaller batch for Mac 8GB RAM
        per_device_eval_batch_size=4,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to=[]  # disables wandb / logging
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
    tokenizer.save_pretrained("../models/distilbert_model")
    model.save_pretrained("../models/distilbert_model")

    # Evaluate DistilBERT
    predictions = trainer.predict(val_dataset)
    distilbert_preds = predictions.predictions.argmax(-1)
    results['distilbert'] = (distilbert_preds == Y_test).mean()

    return results, distilbert_preds
