# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

<img width="822" height="565" alt="image" src="https://github.com/user-attachments/assets/72deff70-4895-45d8-93f5-b0bcd8343241" />



## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:
### Register Number:
### Name:Deepika  R
### Register No:212224040061
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/DL1.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1,8)
        self.fc2=nn.Linear(8,10)
        self.fc3=nn.Linear(10,1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}

  def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
# Write your code here
ai_brain=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.Adam(ai_brain.parameters(),lr=0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = ai_brain(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        ai_brain.history['loss'].append(loss.item())



        ai_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')


train_model(ai_brain, X_train_tensor, y_train_tensor, criterion, optimizer)


with torch.no_grad():
    test_loss = criterion(ai_brain(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')


loss_df = pd.DataFrame(ai_brain.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = ai_brain(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')




```
## Dataset Information
<img width="312" height="371" alt="image" src="https://github.com/user-attachments/assets/3a6b9b0c-3d69-44cf-aeda-f9010976687f" />



## OUTPUT
<img width="290" height="492" alt="image" src="https://github.com/user-attachments/assets/45ff4246-65c0-458d-962f-903b88437aee" />

<img width="806" height="182" alt="image" src="https://github.com/user-attachments/assets/f0d7c852-66d0-45c3-a597-bb8eeab17f30" />


<img width="803" height="20" alt="image" src="https://github.com/user-attachments/assets/639fcd4f-3b76-44a1-afa6-628613584655" />



### Training Loss Vs Iteration Plot

<img width="808" height="627" alt="image" src="https://github.com/user-attachments/assets/072760fc-f355-456d-b71d-006d2a7d7dcf" />





### New Sample Data Prediction

<img width="641" height="34" alt="image" src="https://github.com/user-attachments/assets/7eab9cb7-3bba-41e1-9641-04200df5dc5c" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.

