from django.shortcuts import render,HttpResponse
from . import main,predict as p
# Create your views here.
def check(request):
    return render(request,"HomePage.html")
def form(request):
    return render(request,"Form.html")
def train(request):
    data=main.train_model_with_data()
    print("__________________\n____________________")
    print("__________________\n____________________")
    print(data)
    return render(request,"TrainAndSavePage.html",{"data":data})
def predict(request):
    input=[]
    var=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    for q in var:
        if(q=='oldpeak'):
            input.append(float(request.GET[q]))
        else:
            input.append(int(request.GET[q]))
    print(input)
    input.append(0)# dummy target
    input=main.train_model_with_data(toPredict=True,data_to_predict=input)

    print(type(input))
    predicted_out=p.get_predictions(input)
    res=0
    print(predicted_out)
    for i in predicted_out:
        if(i[1]=='1'):
            res+=1
            i.pop()
            i.append("YES")
        else:
            i.pop()
            i.append("NO")
    res=(100/5)*res
    if(res==0.0):
        color="#b5f777"
    elif(res==20.0):
        color="#77ddf7"
    elif(res==40.0):
        color="#96b1ff"
    elif(res==60.0):
        color="#96b1ff"
    elif(res==80.0):
        color="#fa96ff"
    else:
        color="#f25e5e"
    return render(request,"result.html",{"result":predicted_out,"res":res,"color":color})