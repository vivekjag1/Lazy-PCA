import pandas as pd
import numpy as np
from power_iteration import PowerIteration as nPCA
from lazy_pca import LazyPCA as LPCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# pca = PCA()

NUM_COMPS = 100


df = pd.read_csv('med_data_with_embeds.csv', usecols=['tweet_id','entity','sentiment','embeds'])
raw_embeds = df['embeds'].values
embed_len = len(np.fromstring(raw_embeds[0][1:-1],sep=' '))
embeds = np.zeros((len(raw_embeds),embed_len))
for i in range(len(raw_embeds)):
    embeds[i] = np.fromstring(raw_embeds[i][1:-1],sep=' ')
labels = df['sentiment'].values


pca = PCA(n_components=NUM_COMPS,svd_solver='randomized',random_state=52818)
npca = nPCA(num_components=NUM_COMPS)
lazy = LPCA(theta_thresh=5)

base_xform = pca.fit_transform(embeds)
norm_xform, norm_comps = npca.fit_transform(embeds)
lazy_xform, lazy_comps = lazy.fit_transform(embeds,NUM_COMPS)

# train/test split
lxtrain,lxtest, lytrain, lytest = train_test_split(
    norm_xform, 
    labels, 
    test_size=0.2, 
    random_state=42
)
X_train, X_test, y_train, y_test= train_test_split(
    lazy_xform, 
    labels, 
    test_size=0.2, 
    random_state=42
    )
base_xtrain, base_xtest, base_ytrain, base_ytest = train_test_split(
    base_xform, 
    labels, 
    test_size=0.2, 
    random_state=42
)

#sklearn Gaussian Naive Bayes Classifier,for continuous embedding values
model = GaussianNB()
lmodel = GaussianNB()
bmodel = GaussianNB()
model.fit(X_train, y_train)
lmodel.fit(lxtrain, lytrain)
bmodel.fit(base_xtrain, base_ytrain)

y_pred = model.predict(X_test)
lypred = lmodel.predict(lxtest)
bypred = bmodel.predict(base_xtest)

#toss some evaluation in there
print(f"Normal PCA accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Normal PCA Classification Report:\n{classification_report(y_test, y_pred)}")
print(confusion_matrix(y_test, y_pred))

print(f"Lazy PCA accuracy: {accuracy_score(lytest,lypred)}")
print(f"Lazy PCA Classification Report:\n{classification_report(lytest, lypred)}")
print(confusion_matrix(lytest,lypred))

print(f"Base PCA accuracy: {accuracy_score(base_ytest,bypred)}")
print(f"Lazy PCA Classification Report:\n{classification_report(base_ytest, bypred)}")
print(confusion_matrix(base_ytest,bypred))