from fastai.data.load import DataLoader
from fastai.torch_core import requires_grad
import string
import torch 
import base64
import datetime
import fastai
from PIL import Image
from torch import nn
from fastai import learner

# How 2 Train an AI:
data = torch.FloatTensor([([1,2],[3,6]),([4,8],[5,10]),([3,6],[6,12]),([5,10],[7,14])])
def train_epoch(model, learningrate, params): # Give it a model (Hint: method name to call), an int and a list of parameters to change
    for xb in data:
        calculateGradient(params,xb,model)
        for p in params:
            p.data -=p.grad*learningrate
            p.grad.zero_()
        print(params)

#How to calculate a gradient 
def calculateGradient(xb,yb,model):
    predictions = model(xb) # Predict something using the model
    loss = isThisLoss(predictions, yb) # Compare the model predictions to the actual result
    loss.backward() # Calculate the change in accuracy/variance of the model 

# Simple Neural Network
weights = torch.FloatTensor(([1,1],[1,1]))
weights2 = torch.FloatTensor(([1,1],[1,1]))
bias = 1
def StoopidModel(data2):
    linear = data2@weights + bias # Matrix multiply the data values by the weights for those values, account for potential bias
    print(linear)
    linear = linear.max(torch.tensor(0.0)) # Remove linearity by also setting negative values to 0
    linear = linear@weights + bias # Do it all again with different sets of weights/bias
    return linear

#Simple loss function (I have no fucking clue if this is right or not let's just go with yes)
def isThisLoss(predictions, data):
    return torch.where(data>predictions, predictions+1, predictions-1)
#I fucking swear this is 1000x easier than I thought it would be...
train_epoch(StoopidModel, 0.001, torch.FloatTensor(([2,3],[3,4])))
