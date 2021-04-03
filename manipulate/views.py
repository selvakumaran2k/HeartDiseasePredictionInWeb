from django.shortcuts import render,HttpResponse

# Create your views here.
def check(request):
    return render(request,"Home.html")