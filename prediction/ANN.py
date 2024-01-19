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
def test():
    return 'hello'
def getPredict( age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ):
    #Reading CSV File
    df = pd.read_csv(f'{settings.BASE_DIR}/prediction/pre_data/cancer_patient_data_sets.csv')
    #Low --> 0 ; Medium --> 1 ; High --> 2
    Range_Mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Level'] = df['Level'].map(Range_Mapping)

    #Dividing X and Y
    x = torch.tensor(df.iloc[: , 2:25].values).float()
    y = torch.tensor(df.iloc[: , 25].values).float()

    #Checking If # Low, Medium, High Is Around The Same
    count_0 = torch.sum(torch.eq(y, 0)).item()
    count_1 = torch.sum(torch.eq(y, 1)).item()
    count_2 = torch.sum(torch.eq(y, 2)).item()

    #Dividing Training & Testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #Changing Shape of Y
    y_train = y_train.view(-1).long()
    y_test = y_test.view(-1).long()

    #--------------------------------------------------------------

    #Checking If #Low, Medium, High Is Around The Same -- Training
    count_0 = torch.sum(torch.eq(y_train, 0)).item()
    count_1 = torch.sum(torch.eq(y_train, 1)).item()
    count_2 = torch.sum(torch.eq(y_train, 2)).item()

    #Checking If #Low, Medium, High Is Around The Same -- Testing
    count_0 = torch.sum(torch.eq(y_test, 0)).item()
    count_1 = torch.sum(torch.eq(y_test, 1)).item()
    count_2 = torch.sum(torch.eq(y_test, 2)).item()

    #--------------------------------------------------------------

    


    #loss-function
    lossfunction = nn.CrossEntropyLoss()

    #optimizer
    optimizer = torch.optim.SGD(ANNmodel.parameters() , lr = 0.01)

    #-----------------------------------------------------------------

    Epochs = 10000
    Incoming_Losses = []

    for i in range(Epochs):

      #front-prop
      Results = ANNmodel(x_train)


      #compute losses
      Losses = lossfunction(Results , y_train)
      Incoming_Losses.append(Losses)

      #back prop
      optimizer.zero_grad()
      Losses.backward()
      optimizer.step()

    #----------------------------------------------------------
      
    #TESTING
    Test_Loss = []

    Testing_Results = ANNmodel(x_test)
    Losses = lossfunction(Testing_Results, y_test)
    Test_Loss.append(Losses)

    #Soft-Max Function
    Probabilities = nn.functional.softmax(Testing_Results, dim=1)
    Prediction = torch.argmax(Probabilities, dim=1)
    #print(Prediction)

    #------------------------------------------------------------------

    #Losses
    #print(Incoming_Losses)
    #print(Test_Loss)

    #-------------------------------------------------------------------

    #Checking How Many Wrong
    Incorrect = 0
    for i in Prediction:
      if Prediction[i] != y_test[i]:
        Incorrect+=1

    #print(Incorrect)
    
#--------------------------------------------------------------------
    
    Inputs = torch.tensor([age, gender, airPollution, alcoholUse, dustAllergy, occupationalHazards, geneticRisk, chronicLungDisease, balancedDiet, obesity, smoking, passiveSmoking, chestPain, coughingBlood, fatigue, weightLoss, shortnessBreath, wheezing, swallowingDifficulty, clubbingFinger, frequentCold, dryCough, snoring ]) # noqa
    
    Model_Prediction = ANNmodel(Inputs.view(1, -1).float())
    #Plain Prediction

    #Using Soft-Max Function
    Probabilities = nn.functional.softmax(Model_Prediction, dim=1)
    Prediction = torch.argmax(Probabilities, dim=1)
    
    FILE = f"media/tmp/{uuid4()}.pt"
    torch.save(ANNmodel, FILE)
    return FILE