import pandas as pd
from sklearn.model_selection import train_test_split
from django.conf import settings
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np

path = settings.MEDIA_ROOT + "//" + "uci_heart.csv"
df = pd.read_csv(path)
X = df.iloc[:, :-1].values  # indipendent variable
y = df.iloc[:, -1].values  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=0)

from hmmlearn.hmm import GaussianHMM


def fitHMM(Q, nSamples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))

    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))

    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q, [len(Q), 1]))

    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, samples


def calc_hmm_model():
    path = settings.MEDIA_ROOT + "//" + "uci_heart.csv"
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values  # indipendent variable
    y = df.iloc[:, -1].values  # Dependent variable
    X = np.column_stack([X, y])
    print("fitting to HMM and decoding ...", end="")
    # Make an HMM instance and execute fit
    model = GaussianHMM(n_components=4, covariance_type="diag", n_iter=1000).fit(X)
    # Predict the optimal sequence of internal hidden state
    hidden_states = model.predict(X)
    print("done")
    print("Transition matrix")
    print(model.transmat_)
    print()
    print("Means and vars of each hidden state")

    li = []
    for i in range(model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", model.means_[i])
        print("var = ", np.diag(model.covars_[i]))
        print()

        rslt  = {str(i)+' hidden state': i, "mean": model.means_[i], "var": np.diag(model.covars_[i])}
        li.append(rslt)
    return li



def calc_ann_model():
    print("*" * 25, "Artificail Neural Network")
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    classifier = Sequential()
    classifier.add(Dense(output_dim=13, init='uniform', activation='relu', input_dim=13))
    classifier.add(Dense(output_dim=13, init='uniform', activation='relu'))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(classifier.summary())
    classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print('AI Accuracy:', accuracy)
    precision = precision_score(y_test, y_pred)
    print('AI Precision Score:', precision)
    recall = recall_score(y_test, y_pred)
    print('AI Recall Score:', recall)
    f1score = f1_score(y_test, y_pred)
    print('AI F1-Score:', f1score)
    return accuracy, precision, recall, f1score



def calc_proposed_model():
    print("*" * 25, "Random Forest Classification")
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('RF Accuracy:', accuracy)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print('RF Precision Score:', precision)
    recall = recall_score(y_test, y_pred)
    print('RF Recall Score:', recall)
    f1score = f1_score(y_test, y_pred)
    print('RF F1-Score:', f1score)
    return accuracy, precision, recall, f1score


def calc_support_vector_classifier():
    print("*" * 25, "SVM Classification")
    from sklearn.svm import SVC
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('SVM Accuracy:', accuracy)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print('SVM Precision Score:', precision)
    recall = recall_score(y_test, y_pred)
    print('SVM Recall Score:', recall)
    f1score = f1_score(y_test, y_pred)
    print('SVM F1-Score:', f1score)
    return accuracy, precision, recall, f1score


def calc_j48_classifier():
    print("*" * 25, "j48")
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    model.fit(X_train, y_train)  # Trained wih 80% Data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('j48 Accuracy:', accuracy)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print('j48 Precision Score:', precision)
    recall = recall_score(y_test, y_pred)
    print('j48 Recall Score:', recall)
    f1score = f1_score(y_test, y_pred)
    print('j48 F1-Score:', f1score)
    return accuracy, precision, recall, f1score



def test_user_date(test_features):
    print(test_features)
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    test_pred = model.predict([test_features])
    return test_pred
