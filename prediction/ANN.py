#IMPORTS
# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.onnx
from django.conf import settings

def getPredict( age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ):
    
    Inputs = torch.tensor([age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ]) # noqa
    ANNmodel = nn.Sequential(
        nn.Linear(23 , 64),
        nn.ReLU(),
        nn.Linear(64 , 64),
        nn.ReLU(),
        nn.Linear(64 , 3),
    )

    ANNmodel.load_state_dict( torch.load(f'{settings.BASE_DIR}/media/model.pth') )

    # Loading Test
    # FILE = "model2.pth"
    # torch.save(ANNmodel.state_dict(), FILE)

    Model_Prediction = ANNmodel(Inputs.view(1, -1).float())
    #Plain Prediction

    #Using Soft-Max Function
    Probabilities = nn.functional.softmax(Model_Prediction, dim=1)
    Prediction = torch.argmax(Probabilities, dim=1)
    
    cancer_probability = Probabilities[:, 1].item()  # Extracting the cancer probability
    print(Model_Prediction)
    print(Probabilities)
    print(Prediction)
    print(Prediction.item())
    result = ["Low", "Medium", "High"]
    # choose one among following return values
    return result[ Prediction.item() ]
    return cancer_probability