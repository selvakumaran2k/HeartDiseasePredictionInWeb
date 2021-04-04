import pickle
model_name_list=[
    "models/Linear_regression_model.sav",
    "models/KNearest_neghibours_model.sav",
    "models/Support_Vector_Machine_model.sav",
    "models/Decesion_Tree_Classifier_model.sav",
    "models/RandomForestClassifier_model.sav",
    ]
name=[
    "Linear_regression_model",
    "KNearest_neghibours_model",
    "Support_Vector_Machine_model",
    "Decesion_Tree_Classifier_model",
    "RandomForestClassifier_model",
      ]
models=[]
print("Loading Models......")
for model in model_name_list:
    models.append(pickle.load(open(model,'rb')))
print("Models Loaded.")

def get_predictions(data):
    print("getting in ")
    ans=[]
    n=0
    for model in models:
        print(data.shape)
        ans.append([name[n],str((model.predict(data))[0])])
        n+=1
        print("predicting model....")
    print("Done")
    return ans
