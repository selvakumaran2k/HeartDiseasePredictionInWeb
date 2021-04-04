import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score_conf=[]
name=[
    "Linear_regression_model",
    "KNearest_neghibours_model",
    "Support_Vector_Machine_model",
    "Decesion_Tree_Classifier_model",
    "RandomForestClassifier_model",
      ]

def print_score(clf, X_train, y_train, X_test, y_test, name="",train=True):

    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        # score_conf.append([accuracy_score(y_train, pred) * 100,confusion_matrix(y_train, pred)])

    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        score_conf.append([accuracy_score(y_test, pred)*100, confusion_matrix(y_test, pred),name])

import pickle

model_name_list=[
    "models/Linear_regression_model.sav",
    "models/KNearest_neghibours_model.sav",
    "models/Support_Vector_Machine_model.sav",
    "models/Decesion_Tree_Classifier_model.sav",
    "models/RandomForestClassifier_model.sav",
    ]
def train_models_and_save(X_train,y_train,X_test,y_test,name):
    score_conf.clear()
    #training the model
    # one LogisticRegression
    print("Training model")
    from sklearn.linear_model import LogisticRegression
    lr_clf = LogisticRegression(solver='liblinear')
    lr_clf.fit(X_train, y_train)
    pickle.dump(lr_clf,open(model_name_list[0],'wb'))
    # printing score

    print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(lr_clf, X_train, y_train, X_test, y_test, name[0],train=False)
    #trainng model
    # KNeighborsClassifier
    print("Training model")

    from sklearn.neighbors import KNeighborsClassifier
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    pickle.dump(knn_clf,open(model_name_list[1],'wb'))

    # printing score

    print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(knn_clf, X_train, y_train, X_test, y_test, name[1],train=False)
    #trining model
    # support vector machine
    print("Training model")

    from sklearn.svm import SVC
    svm_clf = SVC(kernel='rbf', gamma=0.1, C=1.0)
    svm_clf.fit(X_train, y_train)
    pickle.dump(svm_clf,open(model_name_list[2],'wb'))

    # printing score

    print_score(svm_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(svm_clf, X_train, y_train, X_test, y_test, name[2],train=False)
    #training model
    # decision tree classifier
    print("Training model")

    from sklearn.tree import DecisionTreeClassifier
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    pickle.dump(tree_clf,open(model_name_list[3],'wb'))

    # printing score
    print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(tree_clf, X_train, y_train, X_test, y_test, name[3],train=False)
    # training model
    # Random forest
    print("Training model")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV

    rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf_clf.fit(X_train, y_train)
    pickle.dump(rf_clf,open(model_name_list[4],'wb'))

    # printing score
    print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
    print_score(rf_clf, X_train, y_train, X_test, y_test, name[4],train=False)
    return score_conf
