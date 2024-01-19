from django.shortcuts import render
from django.http import JsonResponse, FileResponse, HttpResponse
# import ANN
import prediction.ANN as ann

def home(request):
    return render(request, "pages/home.html", {})

def predict(request):
    fileName = ann.getPredict(
        int(request.POST.get('age', 1)),
        int(request.POST.get('gender', 1)),
        int(request.POST.get('airPollution', 1)),
        int(request.POST.get('alcoholUse', 1)),
        int(request.POST.get('dustAllergy', 1)),
        int(request.POST.get('occupationalHazards', 1)),
        int(request.POST.get('geneticRisk', 1)),
        int(request.POST.get('chronicLungDisease', 1)),
        int(request.POST.get('balancedDiet', 1)),
        int(request.POST.get('obesity', 1)),
        int(request.POST.get('smoking', 1)),
        int(request.POST.get('passiveSmoking', 1)),
        int(request.POST.get('chestPain', 1)),
        int(request.POST.get('coughingBlood', 1)),
        int(request.POST.get('fatigue', 1)),
        int(request.POST.get('weightLoss', 1)),
        int(request.POST.get('shortnessBreath', 1)),
        int(request.POST.get('wheezing', 1)),
        int(request.POST.get('swallowingDifficulty', 1)),
        int(request.POST.get('clubbingFinger', 1)),
        int(request.POST.get('frequentCold', 1)),
        int(request.POST.get('dryCough', 1)),
        int(request.POST.get('snoring', 1)),
    )
    data = {
        "result": fileName
    }
    
    return JsonResponse({'file': fileName})