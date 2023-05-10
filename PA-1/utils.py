import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def normal_equation(X, y, l = None):
    if l:
        
        return np.linalg.inv(X.T@X + l*np.identity(X.shape[1]))@X.T@y.reshape(-1,1)

    return (np.linalg.inv(X.T@X)@X.T@y).reshape(-1,1)


def transform(X, degree=1):
    
    if X.shape[1] == 2:
        X_transform = np.ones((X.shape[0], 1))
        if degree == 0:
            return X_transform
        X_transform = np.append(X_transform, X, axis=1)
        

        for i in range(2, degree + 1):
            to_concat = [X_transform]
            for j in range(-i, 0):
                to_concat.append(np.reshape(X_transform[:, j] * X[:, 0], (-1, 1)))
            to_concat.append(np.reshape(X_transform[:, -1] * X[:, 1], (-1, 1)))
            X_transform = np.concatenate(to_concat, axis=1)

    else:
        X_transform = np.ones((X.shape[0], 1))

        for i in range(1, degree + 1):
            X_transform = np.append(X_transform, np.power(X, i).reshape(X.shape[0], X.shape[1]), axis=1)
    
    return X_transform

def loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def grad_update(X, W, y_true, lr, l=None):

    grad = X.T@(X@W - y_true) + l*W if l else X.T@(X@W - y_true)
    
    W -= lr*grad/X.shape[0]
    return W

def plot(x, y, title, xlabel, ylabel, save = None):
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig(save)
    plt.show()

def fun(x, w, degree):
    y = 0
    x = transform(x, degree)
    for i in range(w.shape[0]):
        y += x[:, i]*w[i]
    return y[..., np.newaxis]

def plot_deg(x, y, w, degree, title, xlabel, ylabel, name):
    if x.shape[1] == 1:
        min_, max_ = np.min(x,axis=0)[0], np.max(x, axis=0)[0]
        x_ = np.linspace(min_, max_, num=100)[..., np.newaxis]
        plt.figure(figsize=(12,5))
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.plot(x_, fun(x_, w, degree))
        plt.plot(x, y, "cx")
        plt.savefig(f"./{name}.png")
        plt.show()
        
    else:
        plt.figure()
        min_, max_ = np.min(x, axis=0), np.max(x, axis=0)
        x_ = np.linspace(min_[0], max_[0], num=100)[..., np.newaxis]
        y_ = np.linspace(min_[1], max_[1], num=100)[..., np.newaxis]
        X, Y = np.meshgrid(x_, y_)
        n = X.shape[0]
        Z = fun(np.stack([X.reshape(-1), Y.reshape(-1)], axis=1), w, degree)
        Z = Z.reshape(n, -1)
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50)
        ax.scatter(x[:,0], x[:,1], np.squeeze(y, axis=1))
        ax.set_title(f'surface_{degree}')
        plt.savefig(name)
        plt.show()
