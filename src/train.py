from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

results={}
def train_classical_models(X_train,Y_train,X_test,Y_test):
    import os
    if not os.path.exists("../models"):
        os.makedirs("../models")


    ##Vectorizer##
    vectorizer=TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    ##---------##

    ##Logistic_Regression##
    logistic_model=LogisticRegression()
    logistic_model.fit(X_train_vec,Y_train)
    results['logistic_class']=logistic_model.score(X_test_vec,Y_test)
    ##------------------##

    ##NaiveByes##
    NB_model=MultinomialNB()
    NB_model.fit(X_train_vec,Y_train)
    results['naive_bayes']=NB_model.score(X_test_vec,Y_test)





    joblib.dump(logistic_model, "../models/logistic_class.pkl")
    joblib.dump(NB_model, "../models/naive_bayes.pkl")
    joblib.dump(vectorizer, "../models/tfidf.pkl")

    return results
    
    
if __name__=="__main__":
    results.update(train_classical_models(X_train,Y_train,X_test,Y_test))
    print('final comparison',results)

