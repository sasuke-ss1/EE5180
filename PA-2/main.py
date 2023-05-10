from utils import *
from glob import glob

data = glob("./PA2/*")
data = sorted(data)

X_train1, X_test1, data1 = test_train_split(data[0])
X_train2, X_test2, data2 = test_train_split(data[1])

NaiveBayes = Classifier(Naive=True)
Bayes = Classifier()

NaiveBayes.train(X_train1)
cm = NaiveBayes.predict(X_test1, dataset="1")
params1N = NaiveBayes.get_params()
plot(data1, cm, params1N, "Naive_Bayes_Data1", "feat1", "feat2", "1")

NaiveBayes.train(X_train2)
cm = NaiveBayes.predict(X_test2, dataset="2")
params2N = NaiveBayes.get_params()
plot(data2, cm, params2N, "Naive_Bayes_Data2", "feat1", "feat2", "2")

Bayes.train(X_train1)
cm = Bayes.predict(X_test1, dataset='1')
params1 = Bayes.get_params()
plot(data1, cm, params1 , "Bayes_Data1", "feat1", "feat2", "3")

Bayes.train(X_train2)
cm = Bayes.predict(X_test2, dataset='2')
params2 = Bayes.get_params()
plot(data2, cm, params2, "Bayes_Data2", "feat1", "feat2", "4")
