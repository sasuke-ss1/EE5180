from utils import *
from argparse import ArgumentParser
from tqdm import tqdm

parser= ArgumentParser()
parser.add_argument("--path", "-p", default="./", type=str, help="The root directory path of the data .mat files")

args = parser.parse_args()

def main():
    trainData, trainLabels = loadData(args.path)
    testData, testLabels = loadData(args.path, False)

    trainList = characterize(trainData, trainLabels) 
    testList = characterize(testData, testLabels)
    #Question 1
    visualize(trainList, "train")
    visualize(testList, "test")

    trainMean = mean(trainData)
    testMean = mean(testData)
    
    imagePlot(trainMean, "Train Mean Image", "trainMean")
    imagePlot(testMean, "Test Mean Image", "testMean")

    trainCentered = centered(trainList, trainMean)
    testCentered = centered(testList, trainMean)

    visualize(trainCentered, "trainCentered")
    visualize(testCentered, "testCentered")
    
    #Question 2
    trainEval, trainEvec = eig(trainCentered)
    trainEvec = matrix_norm(trainEvec, axis=1)
    
    visualize(trainEvec[:25, :], "trainFace")
    plot(trainEval, "EigenValue", "num", "Value", "Eval")
    p=0.95
    k = findNumE(trainEval, p)
    print("The value of d for the train set is : ", trainEval.size, "\n")
    print(f"We need {k} components to retain {p*100}% of the data variance")

    #Question 3
    trainConverted = convert(trainCentered, trainEvec[:k, :])
    testConverted = convert(testCentered, trainEvec[:k, :])

    preds, dists, match = predict(trainConverted, testConverted)
    print("The Test Accuracy is :", (preds == testLabels).mean())
    
    bestMatch(trainList, testData, preds, match, "final")

    x, y = [], []
    for k in tqdm(range(1, trainEval.size+1)):
        trainConverted = convert(trainCentered, trainEvec[:k, :])
        testConverted = convert(testCentered, trainEvec[:k, :])

        preds, _, _ = predict(trainConverted, testConverted)
        x.append(k);y.append((preds == testLabels).mean())

    plot([x, y], "AccuracyvsK", "K", "Accuracy", "final", True)


if __name__ == "__main__":
    main()