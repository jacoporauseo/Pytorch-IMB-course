# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 15:23:49 2025

@author: jraus
"""

import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import optim


###### EX 1

# 1. Import the data
data = pd.read_csv("league_of_legends_data_large.csv")

# 2. Define X and y
X = data.drop('win', axis=1)
y = data['win']

# 3. Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Transform to PyTorch tensors
X_train = torch.from_numpy(X_train_scaled).float()
X_test = torch.from_numpy(X_test_scaled).float()

y_train = torch.from_numpy(y_train.to_numpy()).float()
y_test = torch.from_numpy(y_test.to_numpy()).float()


### NOT USED IN MY CODE: in case we wanna use batch SGD

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)



###### EX 2


# 1. Define Logistic model
class LogisticRegressionModel(nn.Module):
    def __init__(self, in_size, out_size = 1):
        super(LogisticRegressionModel,self).__init__()
        self.linear = nn.Linear(in_size,out_size)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out


# 2. Initialize the model
model = LogisticRegressionModel(in_size=X_train.shape[1])

criterion = nn.BCELoss() 

optimizer = optim.SGD(model.parameters(), lr = 0.01)



###### EX 3

# Write your code here
epochs = 1000

for epoch in range(epochs):
    #################################
    # Training mode
    #################################
    model.train()
    y_train = y_train.view(-1,1)
    # Reset gradient
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train)
    # Compute Loss
    loss = criterion(outputs, y_train)
    # Backpropagation
    loss.backward()
    # Update parameters
    optimizer.step()

    if epoch % 10 == 0: 
        print(f"At epoch {epoch} the loss is {loss.item()}")

#################################
# Eval mode
#################################
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    train_outputs = model(X_train)

# Accuracy 
train_preds_class = (train_outputs >= 0.5).float()
test_preds_class = (test_outputs >= 0.5).float()

train_accuracy = (train_preds_class.eq(y_train.view(-1, 1))).float().mean().item()
test_accuracy = (test_preds_class.eq(y_test.view(-1, 1))).float().mean().item()

train_accuracy
test_accuracy


###### EX 4

optimizer1 = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)


# Write your code here
epochs = 1000

for epoch in range(epochs):
    #################################
    # Training mode
    #################################
    model.train()
    y_train = y_train.view(-1,1)
    # Reset gradient
    optimizer1.zero_grad()
    # Forward pass
    outputs = model(X_train)
    # Compute Loss
    loss = criterion(outputs, y_train)
    # Backpropagation
    loss.backward()
    # Update parameters
    optimizer.step()

    if epoch % 10 == 0: 
        print(f"At epoch {epoch} the loss is {loss.item()}")

#################################
# Eval mode
#################################
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    train_outputs = model(X_train)

# Accuracy 
train_preds_class = (train_outputs >= 0.5).float()
test_preds_class = (test_outputs >= 0.5).float()

train_accuracy = (train_preds_class.eq(y_train.view(-1, 1))).float().mean().item()
test_accuracy = (test_preds_class.eq(y_test.view(-1, 1))).float().mean().item()

train_accuracy
test_accuracy




#### Ex5

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, 
roc_curve, auc
import matplotlib.pyplot as plt


# Switch to evaluation mode
model.eval()
with torch.no_grad():
    test_probs = model(X_test).view(-1)  # predicted probabilities
    test_preds = (test_probs >= 0.5).int()  # binary predictions

# Convert to NumPy for sklearn
y_true = y_test.numpy()
y_pred = test_preds.numpy()
y_scores = test_probs.numpy()

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n")
print(classification_report(y_true, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()


####### EX 6

# Save the model
torch.save(model.state_dict(), 'jr.pth')

# Load the model
model2 = LogisticRegressionModel(in_size=X_train.shape[1])
model2.load_state_dict(torch.load('jr.pth'))

# Ensure the loaded model is in evaluation mode
model2.eval() 
with torch.no_grad():
    out = model2(X_test)
    preds = (out >= 0.5).int()  


#### EX7

epochs = 100
learn_rate = [0.0001, 0.001, 0.01, 0.1]

test_accuracies = {}


for lr in learn_rate:
    print(f"\nTraining with learning rate = {lr}")
    
    # Model
    model = LogisticRegressionModel(in_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        yhat = model(X_train).view(-1)
        loss = criterion(yhat,y_train.view(-1).float())
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_probs = model(X_test).view(-1)
        test_preds = (test_probs >= 0.5).int()
        accuracy = (test_preds == y_test).sum().item() / len(y_test)
        test_accuracies[lr] = accuracy
        print(f"Test Accuracy @ LR={lr}: {accuracy:.4f}")

print(test_accuracies) # 0.01 has the highest accuracy: 0.51




##### Ex.8

w = model.linear.weight.data.view(-1).numpy()

# Use first exercice pd df columns names
names = X.columns

df = pd.DataFrame({
    'Feature': names,
    'Weight': w
})

df.sort_values(by = 'Weight', ascending=False)

fig, ax = plt.subplots()
plt.bar(df.Feature, df.Weight)
ax.set_ylabel('Weigth')
ax.set_title('Feature Importance')


plt.show()


# gold_earned is the most important feature, cs the least important
