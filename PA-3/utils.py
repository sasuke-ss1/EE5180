from scipy.io import loadmat
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import random



def loadData(path: str, train=True):
    path = os.path.join(path, "train_set.mat") if train else os.path.join(path, "test_set.mat")
    ret = loadmat(path)

    return ret["data"], np.squeeze(ret["labels"], 0)

def makeImage(array, h, w):

    return np.reshape(array, (h, w, 1))

def makeDict(array, h=112, w=92):
    ret = {}

    for i in range(array.shape[0]):
        ret[i] = [makeImage(array[i], h, w)]

    return ret

def characterize(data: np.ndarray, labels: np.ndarray, h=112, w=92):  
    l = np.unique(labels)
    ret = {i:[] for i in l}

    for i, label in enumerate(labels):
        ret[label].append(makeImage(data[i], h, w))

    return ret


def mean(arr: np.ndarray, h=112, w=92):
    array = arr.copy()
    return np.mean(array, axis=0).reshape(h, w, 1)

def visualize(data: dict, name=""):
    if type(data) != dict:
        data = makeDict(data)
        fig, axs = plt.subplots(5, 5, figsize=(10,10))
        
        for i in range(25):
            img = random.choice(data[i])

            axs[i//5, i%5].set_title(f"Egf: {i}")
            axs[i//5, i%5].imshow(img)
            axs[i//5, i%5].set_xticks([])
            axs[i//5, i%5].set_yticks([])

        fig.supylabel("Visualization")
        
        if name:
            plt.savefig(name + ".png")

        plt.show();return
    

    fig, axs = plt.subplots(5, 8, figsize=(10,10))
    
    for i in range(40):
        img = random.choice(data[i])

        axs[i//8, i%8].set_title(f"Class: {i}")
        axs[i//8, i%8].imshow(img)
        axs[i//8, i%8].set_xticks([])
        axs[i//8, i%8].set_yticks([])

    fig.supylabel("Visualization")
    
    if name:
        plt.savefig(name + ".png")

    plt.show() 

def centered(dat: dict, mean: np.ndarray):
    data = dat.copy()
    for label in data.keys():
        data[label] = list(map(lambda x: x-mean, data[label]))
    
    return data

def flatten(dat: dict):
    data = dat.copy()
    for label in data.keys():
        data[label] = list(map(lambda x: x.reshape(-1,), data[label]))
    
    return data

def collapse(data: dict):
    ret = []
    flat = flatten(data)
    for val in flat.values():
        for v in val:
            ret.append(v)

    return np.array(ret)

def eig(data: dict):
    collapsed = collapse(data.copy())
    sym = collapsed@collapsed.T/(collapsed.shape[0])

    eign, u = np.linalg.eig(sym)

    idx = eign.argsort()[::-1]
    eign = eign[idx]
    u = u[:,idx]

    return eign, u.T@collapsed

def plot(array: np.ndarray, title: str, xlabel: str, ylabel: str, name: str, xy=False):
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.plot(array[0], array[1]) if xy else plt.plot(array)

    plt.savefig(f"{name}.png")
    plt.show()

def imagePlot(img: np.ndarray, title: str, name: str, h=112, w=92):
    plt.figure(figsize=(12, 5))
    plt.title(title)
    
    plt.imshow(img.reshape(h, w, 1))

    plt.savefig(f"{name}.png")
    plt.show()


def findNumE(Eval: np.ndarray, percent: float):
    i, tmp, den = 0, 0, Eval.sum()
    while(tmp<percent):
        tmp += Eval[i]/den;i+=1
    
    return i

def convert(X: dict, U: np.ndarray):
    collapsed = collapse(X.copy())
    
    return (U@collapsed.T).T

def findClosest(train: np.ndarray, test: np.ndarray):

    bestDist = np.linalg.norm((train[0, :] - test))
    bestI = 0
    bestMatch = None
    for idx, a in enumerate(train[1:, :]):
        if bestDist > np.linalg.norm(a-test):
            bestDist = np.linalg.norm(a-test)
            bestI = idx+1
            bestMatch = a

    return bestI//8, bestDist, bestI%8    

def predict(train: np.ndarray, test: np.ndarray):
    preds, dist, match = [], [], []
    for sample in test:
        pred = findClosest(train, sample)
        preds.append(pred[0])
        dist.append(pred[1])
        match.append(pred[2])
    
    return np.array(preds), np.array(dist), np.array(match)

def bestMatch(train: dict, test: np.ndarray, preds: np.ndarray, match: np.ndarray, name:str, h=112, w=92):
    fig, axs = plt.subplots(10, 16, figsize=(25, 20))
    for i in range(10):
        for j in range(8):
            img = test[8*i+j]
            axs[i][2*j].imshow(makeImage(img, h, w)) 
            axs[i][2*j+1].imshow(train[preds[8*i+j]][match[8*i+j]])
            axs[i][2*j].set_xticks([])
            axs[i][2*j+1].set_xticks([])
            axs[i][2*j].set_yticks([])
            axs[i][2*j+1].set_yticks([])
    for j in range(8):
        axs[0][2*j].set_title(f"T")
        axs[0][2*j+1].set_title(f"C")
    
    plt.savefig(name+"1.png")
    plt.show()

def matrix_norm(mat: np.ndarray, axis):
    norm = np.linalg.norm(mat, axis=axis)[..., np.newaxis]
    return mat/norm

if __name__ == "__main__":
    a = np.random.rand(320, 15)
    b = np.random.rand(15,)
    print(findClosest(a, b))
    