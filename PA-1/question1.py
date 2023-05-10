import numpy as np
import pandas as pd
from utils import normal_equation, transform, loss, plot, plot_deg
import matplotlib.pyplot as plt
import sys



def solve(X_train, y_train,X_val, y_val, degree, name=None):
    loss_val = []
    prev_loss = 1000000
    best_deg = None
    
    
    for i in degree:
        X_train_t = transform(X_train, i)
        
        W = normal_equation(X_train_t, y_train)
        X_val_t = transform(X_val, i)
        y_pred_val = X_val_t@W
        curr_loss = loss(y_pred_val, y_val)
        loss_val.append(curr_loss)
        if curr_loss <= prev_loss:
            best_deg = i
            prev_loss = curr_loss
        plot_deg(X_train, y_train, W, i, f"Degree:{i}", "x", "y", f"degree_{i}_{X_train.shape[1]}")
    #print(loss_val)
    plot(degree, loss_val, "Degree vs Val_loss", "Degree of Polynomial", "Validation Loss", name)

    return best_deg


def solve2(degree, X_train, y_train, X_val, y_val, l_min=100, lmax=200, name=None):
    loss_val = []
    la = np.linspace(l_min, lmax, num=500)
    for l in la:
        X_train_t = transform(X_train, degree)
        W = normal_equation(X_train_t, y_train, l)
        X_val_t = transform(X_val, degree)
        
        y_pred_val = X_val_t@W
        loss_val.append(loss(y_pred_val, y_val))
    plot(la, loss_val, "Lambda vs Val_loss", "Lambda", "Validation Loss", name)


if __name__ == "__main__":
    df1 = pd.read_csv("./data1.txt", delim_whitespace=True, header=None)
    df2 = pd.read_csv("./data2.txt", delim_whitespace=True, header=None)

    data1 = df1[df1.columns].to_numpy()
    data2 = df2[df2.columns].to_numpy()
    

    X_train1 = data1[:int(0.7*data1.shape[0]), :-1]
    y_train1 = data1[:int(0.7*data1.shape[0]), -1].reshape(-1,1)
    X_val1 = data1[int(0.7*data1.shape[0]): int(0.9*data1.shape[0]), :-1]
    y_val1 = data1[int(0.7*data1.shape[0]): int(0.9*data1.shape[0]), -1].reshape(-1,1)
    X_test1 = data1[int(0.9*data1.shape[0]): , :-1]
    y_test1 = data1[int(0.9*data1.shape[0]): , -1].reshape(-1,1)

    X_train2 = data2[:int(0.7*data2.shape[0]), :-1]
    y_train2 = data2[:int(0.7*data2.shape[0]) , -1].reshape(-1,1)
    X_val2 = data2[int(0.7*data2.shape[0]): int(0.9*data2.shape[0]), :-1]
    y_val2 = data2[int(0.7*data2.shape[0]): int(0.9*data2.shape[0]), -1].reshape(-1,1)
    X_test2 = data2[int(0.9*data2.shape[0]): , :-1]
    y_test2 = data2[int(0.9*data2.shape[0]): , -1].reshape(-1,1)




    degree = list(range(0, 11))
    best_deg1 = solve(X_train1, y_train1, X_val1, y_val1, degree, "./1.png")
    best_deg2 = solve(X_train2, y_train2, X_val2, y_val2, degree, "./2.png")

    print(f"Best train fit for 1 feature: {best_deg1}")
    print(f"best train fit for 2 features: {best_deg2}")

    solve2(10, X_train1, y_train1, X_val1, y_val1, name="./3.png")
    solve2(10, X_train2, y_train2, X_val2, y_val2, name="./4.png")
