import pandas as pd
import numpy as np
from power_iteration import PowerIteration as nPCA
from lazy_pca import LazyPCA as LPCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# pca = PCA()

NUM_COMPS = 20


df = pd.read_csv('med_data_with_embeds.csv', usecols=['tweet_id','entity','sentiment','embeds'])
raw_embeds = df['embeds'].values
embed_len = len(np.fromstring(raw_embeds[0][1:-1],sep=' '))
embeds = np.zeros((len(raw_embeds),embed_len))
for i in range(len(raw_embeds)):
    embeds[i] = np.fromstring(raw_embeds[i][1:-1],sep=' ')
labels = df['sentiment'].values


for ang in range(1,6):
    results = {
        "number of components":[],
        "base accuracy":[],
        "sklearn accuracy":[],
    }
    for i in range(5,101,5):
        NUM_COMPS = i
        # first get the transformed data for each PCA method

        if ang==1:
            # only do normal and sklearn PCA once
            pca = PCA(n_components=NUM_COMPS,svd_solver='randomized',random_state=52818)
            npca = nPCA(num_components=NUM_COMPS)
            sk_xform = pca.fit_transform(embeds)
            norm_xform, norm_comps = npca.fit_transform(embeds)
            # normal PCA sets
            X_train, X_test, y_train, y_test= train_test_split(
                norm_xform,
                labels, 
                test_size=0.2, 
                random_state=42
                )
            # sklearn PCA sets
            sk_xtrain, sk_xtest, sk_ytrain, sk_ytest = train_test_split(
                sk_xform, 
                labels, 
                test_size=0.2, 
                random_state=42
            )
            skmodel = GaussianNB()
            model = GaussianNB()
            skmodel.fit(sk_xtrain, sk_ytrain)
            model.fit(X_train, y_train)
            sk_ypred = skmodel.predict(sk_xtest)
            y_pred = model.predict(X_test)
            sk_acc = accuracy_score(sk_ytest,sk_ypred)
            base_acc = accuracy_score(y_test,y_pred)
            results['sklearn accuracy'].append(sk_acc)
            results['base accuracy'].append(base_acc)
            results['number of components'].append(NUM_COMPS)




        
        lazy = LPCA(theta_thresh=ang)

        lazy_xform, lazy_comps = lazy.fit_transform(embeds,NUM_COMPS)

        # then get the train/test splits of projected embeddings
        # lazy PCA sets
        lxtrain,lxtest, lytrain, lytest = train_test_split(
            lazy_xform, 
            labels, 
            test_size=0.2, 
            random_state=42
        )
        

        #sklearn Gaussian Naive Bayes Classifier,for continuous embedding values
        lmodel = GaussianNB()

        lmodel.fit(lxtrain, lytrain)

        lypred = lmodel.predict(lxtest)

        lazy_acc = accuracy_score(lytest,lypred)
        

        results[f"{ang} acc"].append(lazy_acc)

        print(f"{ang} angle, {NUM_COMPS} components finished with accuracies:")
        print(f"{round(lazy_acc*100,2)} lazy\t ")

    res_df = pd.DataFrame(results)

    res_df.to_csv(f"med_good_results.csv")




