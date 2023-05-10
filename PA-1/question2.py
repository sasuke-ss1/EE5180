import numpy as np
import pandas as pd
from utils import normal_equation, grad_update, transform, plot, loss
from tqdm import tqdm
import matplotlib.pyplot as plt


dfT = pd.read_csv("./Train.csv", header=None)
dataT = dfT[dfT.columns].to_numpy()

X_train = dataT[:int(0.8*dataT.shape[0]), :-1]
X_train = transform(X_train)
y_train = dataT[:int(0.8*dataT.shape[0]), -1].reshape(-1,1)

X_val = dataT[int(0.8*dataT.shape[0]): , :-1]
X_val = transform(X_val)
y_val = dataT[int(0.8*dataT.shape[0]): , -1].reshape(-1,1)

dfV = pd.read_csv("./test.csv", header=None)
dataV = dfV[dfV.columns].to_numpy()

X_test = dataV[:, :-1]
X_test = transform(X_test)
y_test = dataV[:, -1].reshape(-1,1)

W_ML = normal_equation(X_train, y_train)
W = np.zeros_like(W_ML)

it = 5000
lr = 0.01

x = list(range(it))
y = []
for i in tqdm(range(it)):
    W = grad_update(X_train, W, y_train, lr)
    y.append(np.sum((W - W_ML)**2))

plot(x, y, "Convergence vs Iterations", "Iterations", "Convergence", "./6.png")
y.clear()
la = np.linspace(0, 100, num=100)

best = []
prev_loss = 10000

for l in tqdm(la):
    W = np.zeros_like(W_ML)
    for i in range(it):
        W = grad_update(X_train, W ,y_train, lr, l)
    y_pred = X_val@W
    curr_loss = loss(y_pred, y_val)
    y.append(curr_loss)
    if curr_loss<= prev_loss:
        best.clear()
        best.append(l);best.append(W)
        #print(l)
        prev_loss = curr_loss
    

assert len(best) != 0, "Didnt Converge, try increasing prev_loss 10 times"    
plot(la,y, "Val_loss vs Lambda", "Lambda", "Val_loss", "./5.png")

print("\n\n")
print(f"Best value for Lambda: {best[0]}")
y_predR = X_test@best[1]
y_predML = X_test@W_ML

print(f"Test_loss with W_R is: {loss(y_predR, y_test)}\n")
print(f"Test_loss with W_ML is : {loss(y_predML, y_test)}")