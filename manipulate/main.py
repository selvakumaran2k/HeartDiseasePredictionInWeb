import pandas as pd
from . import TrainModel
name=[
    "Linear_regression_model",
    "KNearest_neghibours_model",
    "Support_Vector_Machine_model",
    "Decesion_Tree_Classifier_model",
    "RandomForestClassifier_model",
      ]

def train_model_with_data(toPredict=False,data_to_predict=[]):
    import data
    df=data.get_data()
    categorical_val=[]
    if(toPredict):
        df.loc[len(df.index)] = data_to_predict
        print("adding data")
    for column in df.columns:
        if(df[column].unique().shape[0]<=10):
            categorical_val.append(column)
    categorical_val.remove("target")
    from matplotlib import pyplot as plt
    dataset = pd.get_dummies(df, columns = categorical_val)
    from sklearn.preprocessing import StandardScaler

    s_sc = StandardScaler()
    col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])
    if(toPredict):
        dataset.drop('target',axis=1,inplace=True)
        return dataset.loc[len(df)-1:,:]

    # test train splitting
    from sklearn.model_selection import train_test_split
    X = dataset.drop('target', axis=1)
    y = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return TrainModel.train_models_and_save(X_train,y_train,X_test,y_test,name)



#
# from . import predict,TrainModel
# TrainModel.train_models_and_save(X_train,y_train,X_test,y_test)
# sampleData=X_test[:10]
# l=predict.get_predictions(sampleData)
# for ans in l:
#     print(ans)