# Text-Classification-used-Multinomial-Naive-Bayes-model

This program is to show how to build a Multinomial Naive Bayes model to classify text with 2 classes.
This program was done based on Python3 using Machine Leaning library Scikit-learn.

**Program Steps**
- First apply grid search with 5-fold cross validation to find the best values for parameters min_df, stop_words, and alpha of Naive Bayes model.
- Use f1-macro as the scoring metric to select the best parameter values.
- Train a Multinomial Naive Bayes classifier with all samples in the training file by using the best parameter values.
- Test the classifier using the test file. 
- Report the testing performance as:
  + Precision, recall, and f1-score of each label 
  + Treat label 2 as the positive class, plot precision-recall curve and ROC curve, and calculate AUC.
