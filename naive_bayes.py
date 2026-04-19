import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X = ('data_with_embeds.csv')

#train/test split
X_train, X_test, y_train, y_test= train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=21
    )

#sklearn Gaussian Naive Bayes Classifier,for continuous embedding values
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#toss some evaluation in there
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))