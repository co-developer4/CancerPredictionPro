#IMPORTS
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.onnx
from uuid import uuid4
from django.conf import settings

#The Model
ANNmodel = nn.Sequential(
    nn.Linear(23 , 64),
    nn.ReLU(),
    nn.Linear(64 , 64),
    nn.ReLU(),
    nn.Linear(64 , 3),
)
def getPredict( age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ):
    
    Inputs = torch.tensor([age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ]) # noqa
    
    Model_Prediction = ANNmodel(Inputs.view(1, -1).float())
    #Plain Prediction

    #Using Soft-Max Function
    Probabilities = nn.functional.softmax(Model_Prediction, dim=1)
    Prediction = torch.argmax(Probabilities, dim=1)
    
    return Prediction