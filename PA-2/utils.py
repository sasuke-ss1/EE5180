import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

class Classifier():
    def __init__(self, Naive=False):
        self.__X = None
        self.classPrior = {}
        self.__y = None
        self.__u = {}
        self.__var = {}
        self.__Naive = Naive

    def __classPrior(self):

        for cl in self.__classes:    
            cnt = sum(self.__y == cl)
            self.classPrior[cl] = cnt/len(self.__y)

    def __classConditional(self):
        cl = np.unique(self.__y)
       
        
        for c in cl:
            X = self.__X[self.__X[..., -1] == c][:, :-1]
            self.__u[c] = np.mean(X, axis=0)[np.newaxis, ...]
            self.__var[c] = np.diag(np.diag(np.cov(X.T, rowvar=True))) if self.__Naive else np.cov(X.T, rowvar=True)
            print(self.__var[c])
        
    @staticmethod
    def __gaussian(x, mean, var):
        return 1/np.sqrt(np.power(2*np.pi, x.shape[1])*np.linalg.det(var)) * np.exp(-0.5*((x-mean) * (np.linalg.inv(var)@(x-mean).T).T).sum(-1))            

    def __posterior(self, X_test):
        mat = np.zeros((len(self.__classes), X_test.shape[0]))
        for i, cl in enumerate(self.__classes):
            mat[i] = self.classPrior[cl]*self.__gaussian(X_test[..., :-1], self.__u[cl], self.__var[cl])
            
        return mat
                        
    def train(self, X):
        self.__X = X
        self.__y = X[..., -1]
        self.__classes = np.unique(self.__y)

        self.__classPrior()
        self.__classConditional()

    def predict(self, X_test, retP=False, label=True, dataset=""):
        if not label:
            X_test = np.c_[X_test, np.zeros((X_test.shape[0], 1))]

        mat = self.__posterior(X_test)

        if retP:
            rowSum = np.sum(mat.T, axis=1)[..., np.newaxis]
            return mat.T/rowSum
        
        a = np.argmax(mat, axis=0)
        pred = self.__classes[a]

        tmp = "bayes classifier" if not self.__Naive else "naive bayes classifier"
        print(f"Final Validation Accuracy on data_{dataset} using {tmp} was {np.sum(X_test[..., -1] == pred)/X_test.shape[0]}") 
        cm = confusion_matrix(X_test[..., -1], pred)
        
        return cm

    def get_params(self):
        return {"u": self.__u, "cov": self.__var, "prior": self.classPrior}


def confusion_matrix(true, pred):
    K = len(np.unique(true)) 
    result = np.zeros((K, K))
    for i in range(len(true)):
        result[int(true[i]-1)][int(pred[i]-1)] += 1

    return result

def test_train_split(data_path, random_state=42):
    X = pd.read_csv(data_path)
    
    
    X_train = X.sample(frac=0.8, random_state=random_state)
    X_test = X.drop(X_train.index)

    return X_train.to_numpy(), X_test.to_numpy(), X.to_numpy()

def fun(x, y, sigma, mu, prior):    
    sigma[1] = np.linalg.inv(sigma[1])
    sigma[2] = np.linalg.inv(sigma[2])
    a = prior[1] * 1/(2*np.pi*np.sqrt(np.linalg.det(np.linalg.inv(sigma[1])))) * np.exp(-0.5*((x-mu[1][0][0])*(sigma[1][0, 0]*(x - mu[1][0][0]) + sigma[1][0, 1]*(y - mu[1][0][1])) + (y - mu[1][0][1])*(sigma[1][1,0]*(x - mu[1][0][0]) + sigma[1][1,1]*(y - mu[1][0][1]))))
    b = prior[2] * 1/(2*np.pi*np.sqrt(np.linalg.det(np.linalg.inv(sigma[2])))) * np.exp(-0.5*((x-mu[2][0][0])*(sigma[2][0, 0]*(x - mu[2][0][0]) + sigma[2][0, 1]*(y - mu[2][0][1])) + (y - mu[2][0][1])*(sigma[2][1,0]*(x - mu[2][0][0]) + sigma[2][1,1]*(y - mu[2][0][1]))))

    return b/(a + b)

def plot(data, cm, params, title, xlabel, ylabel, name):
    plt.figure(figsize=(12, 5))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    classes = np.unique(data[..., -1])
    X_1 = data[data[..., -1] == classes[0]][:, :-1]
    X_2 = data[data[..., -1] == classes[1]][:, :-1]

    t = np.linspace(-6, 6, 1000)
    h = np.linspace(-6, 6, 1000)
    w_0, w_1 = np.meshgrid(t, h)
    

    z1 = stats.multivariate_normal(params["u"][classes[0]][0],params["cov"][classes[0]]).pdf(np.dstack((w_0, w_1)))
    z2 = stats.multivariate_normal(params["u"][classes[1]][0],params["cov"][classes[1]]).pdf(np.dstack((w_0, w_1)))
    
    
    c1 = plt.scatter(X_1[:, 0], X_1[:, 1], marker="x")
    c2 = plt.scatter(X_2[:, 0], X_2[:, 1], marker="o")
    c3 = plt.contour(t, h, z1)
    c4 = plt.contour(t, h, z2)
    plt.legend((c1, c2), (f"Data_{classes[0]}", f"Data_{classes[1]}"))
    plt.savefig(f"./{name}_contours.png")
    c6 = plt.contour(t, h, fun(w_0, w_1, params["cov"], params["u"], params["prior"]), [0.5], colors=['red']) 
    plt.savefig(f"./{name}.png")
    plt.show()

    df_cm = pd.DataFrame(cm, index = range(1, len(cm)+1), columns = range(1, len(cm) + 1))
    plt.figure(figsize=(12,12))
    ax = plt.subplot()
    tmp = sns.heatmap(df_cm, annot=True)
    ax.set_xlabel("Predicted Labels");ax.set_ylabel('True Labels')
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels([1,2]);ax.yaxis.set_ticklabels([1,2])
    fig = tmp.get_figure()
    fig.savefig(f"./confusion{name}.png", dpi=400)
    plt.show()

