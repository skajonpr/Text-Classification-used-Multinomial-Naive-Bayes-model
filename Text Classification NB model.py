import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,precision_recall_curve
import numpy as np
from matplotlib import pyplot as plt



# Define a function named Classify to take csv file paths of testing and training dataset.
def classify(training_file, testing_file):
    
    # import testing and training data
    test_data=pd.read_csv(testing_file,header=0)
    train_data=pd.read_csv(training_file,header=0)
    # define a pipeline for Gridsearch using Multinomial Naive Bayes method.
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    # define parameters for Gridsearch.
    parameters = {'tfidf__min_df':[1, 2, 5],
                'tfidf__stop_words':[None,"english"],
                'clf__alpha': [0.5, 1.0, 2, 3, 4]}
    # Define the result metric.
    metric =  "f1_macro"
    # Call Gridsearch function with CV = 5.
    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)
    
    # Traning model to get optimal parameters.
    gs_clf = gs_clf.fit(train_data["text"], train_data["label"])
    
    # print out optimal paeameters.
    for param_name in gs_clf.best_params_:
        print("{} : {}".format(param_name , gs_clf.best_params_[param_name]))
        print("best f1 score:", gs_clf.best_score_)
    
    # Store optimal paerameters.
    clf_alpha = gs_clf.best_params_["clf__alpha"]
    tfidf_min_df = gs_clf.best_params_["tfidf__min_df"]
    tfidf_stop_words = gs_clf.best_params_["tfidf__stop_words"]
    
    # Create a pipelne for classification process using Naive Bayes Method.
    classifier = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=tfidf_stop_words,\
                              min_df=tfidf_min_df)),
    ('clf', MultinomialNB(alpha = clf_alpha ))])
    # Traning model.
    clf = classifier.fit(train_data["text"],  train_data["label"])
    # Get Labels
    labels= sorted(train_data["label"].unique())
    labels = list(map(str, labels))
    
    # Predict data used testing dataset.
    predicted = classifier.predict(test_data["text"])
    
    # Print out Classification report (recall, precision, f1-score )
    print(classification_report(test_data["label"], predicted, target_names=labels))
    
    # Get class "2" proabilities.
    predict_p = clf.predict_proba(test_data["text"])
    y_pred = predict_p[:,1]
    # Store labels in an array where 1 is label = class 2.
    binary_y = np.where(test_data["label"] ==2 ,1 ,0)
    # Call roc_curve function and store fpr, tpr, and thresholds data.
    fpr, tpr, thresholds = roc_curve(binary_y, y_pred, pos_label=1)
    # print out ROC graph
    plt.figure();
    plt.plot(fpr, tpr, color='red');
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--');
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate');
    plt.title('AUC of Naive Bayes Model');
    plt.show();
    plt.savefig("AUC.png")
    # Call precision_recall_curve function and store data.
    precision, recall, thresholds = precision_recall_curve(binary_y, y_pred, pos_label=1)
    # Create Precision_Recall_Curve of Naive Bayes Model
    plt.figure();
    plt.plot(recall, precision, color='green', lw=2);
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05]);
    plt.xlabel('Recall');
    plt.ylabel('Precision');
    plt.title('Precision_Recall_Curve of Naive Bayes Model');
    plt.show();
    plt.savefig("Precision_Recall_Curve.png")
    # Print AUC 
    print( "AUC = {}".format(auc(fpr, tpr)))
    
classify("train.csv", "test.csv")