import numpy as np
from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import time
import pickle
from CDNN import CDNN

from draw_paper_pics import make_confusion_matrix_true


def test_iris(k):
    print("---------------Iris dataset------------------")
    print("Loading data.....")
    data = datasets.load_iris()
    n_classes = len(np.unique(data.target))
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    data_dim = len(X_train[0])
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))
    # predict with cdnn
    predict = []
    start_cdnn = time.time()
    for i in range(len(X_test)):
        predict_item = CDNN(X_train, y_train, X_test[i], k)
        predict.append(predict_item)
    t = time.time() - start_cdnn
    acc = accuracy_score(y_test, predict)
    print("Predict time for CDNN: %.3fs" % (t))
    print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

    # predict with sklearn knn
    for weights in ['uniform', 'distance']:
        knn = neighbors.KNeighborsClassifier(k, weights=weights)
        start_knn = time.time()
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        t = time.time() - start_knn
        acc = accuracy_score(y_test, predict)
        print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
        print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

    print("-----------------------------------------------\n")


def test_digits(k):
    print("---------------Digits dataset------------------")
    print("Loading data.....")
    data = datasets.load_digits()
    n_classes = len(np.unique(data.target))
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)
    data_dim = len(X_train[0])
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))
    # predict with cdnn
    predict = []
    start_cdnn = time.time()
    for i in range(len(X_test)):
        predict_item = CDNN(X_train, y_train, X_test[i], k)
        predict.append(predict_item)
    t = time.time() - start_cdnn
    acc = accuracy_score(y_test, predict)
    print("Predict time for CDNN: %.3fs" % (t))
    print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

    # predict with sklearn knn
    for weights in ['uniform', 'distance']:
        knn = neighbors.KNeighborsClassifier(k, weights=weights)
        start_knn = time.time()
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        t = time.time() - start_knn
        acc = accuracy_score(y_test, predict)
        print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
        print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

    print("-----------------------------------------------\n")


def test_breast_cancer(k):
    print("---------------Breast Cancer dataset------------------")
    print("Loading data.....")
    data = datasets.load_breast_cancer()
    n_classes = len(np.unique(data.target))
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

    data_dim = len(X_train[0])
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))
    # predict with cdnn
    predict = []
    start_cdnn = time.time()
    for i in range(len(X_test)):
        predict_item = CDNN(X_train, y_train, X_test[i], k)
        predict.append(predict_item)
    t = time.time() - start_cdnn
    acc = accuracy_score(y_test, predict)
    print("Predict time for CDNN: %.3fs" % (t))
    print("Accuracy for CDNN with k = %d: %.3f\n" % (k, acc))

    # predict with sklearn knn
    for weights in ['uniform', 'distance']:
        knn = neighbors.KNeighborsClassifier(k, weights=weights)
        start_knn = time.time()
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        t = time.time() - start_knn
        acc = accuracy_score(y_test, predict)
        print("Predict time for kNN with %s weights: %.3fs" % (weights, t))
        print("Accuracy for kNN with k = %d and %s weights: %.3f\n" % (k, weights, acc))

    print("-----------------------------------------------\n")


def test_candy(k):
    preds, y_test_dict = {}, {}
    print("---------------Candy dataset------------------")
    print("Loading data.....")
    with open("/home/nfulab/Users/wang/candy/intermediate_result.p", "rb") as f:
        data = pickle.load(f)
    n_classes = 4
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = data['train']["x"], data["val"]["x"], data["train"]["y"], data["val"]['y']
    data_dim = len(X_train[0])
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))
    # predict with cdnn
    predict = []
    start_cdnn = time.time()
    for i in range(len(X_test)):
        predict_item = CDNN(X_train, y_train, X_test[i], k)
        predict.append(predict_item)
    t = (time.time() - start_cdnn)/len(y_test)
    acc = accuracy_score(y_test, predict)
    print("Average Predict time for CDNN: %.4fs" % (t))
    print("Accuracy for CDNN with k = %d: %.4f\n" % (k, acc))
    preds["CDNN"], y_test_dict["CDNN"] = predict, y_test
    # predict with sklearn knn
    for weights in ['uniform', 'distance']:
        knn = neighbors.KNeighborsClassifier(k, weights=weights)
        start_knn = time.time()
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        t = (time.time() - start_knn)/len(y_test)
        acc = accuracy_score(y_test, predict)
        preds[f"KNN with {weights} weights"], y_test_dict[f"KNN with {weights} weights"] = predict, y_test
        print("Predict time for kNN with %s weights: %.4fs" % (weights, t))
        print("Accuracy for kNN with k = %d and %s weights: %.4f\n" % (k, weights, acc))

    print("-----------------------------------------------\n")
    return preds, y_test_dict


def test_candy_svm():
    print("---------------Candy dataset------------------")
    print("Loading data.....")
    with open("/home/nfulab/Users/wang/candy/intermediate_result.p", "rb") as f:
        data = pickle.load(f)
    n_classes = 4
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = data['train']["x"], data["val"]["x"], data["train"]["y"], data["val"]['y']
    data_dim = len(X_train[0])
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9)  # 50%*10% = 5%
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)  # 7 : 3
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))

    # predict with sklearn svm

    svm = SVC()
    start_knn = time.time()
    svm.fit(X_train, y_train)
    predict = svm.predict(X_test)
    t = (time.time() - start_knn)/len(y_test)
    acc = accuracy_score(y_test, predict)
    print("Predict time for svm: %.4f\n" % t)
    print("Accuracy for svm: %.4f\n" % acc)

    print("-----------------------------------------------\n")
    return predict, y_test


def test_candy_random_forest():
    print("---------------Candy dataset------------------")
    print("Loading data.....")
    with open("/home/nfulab/Users/wang/candy/intermediate_result.p", "rb") as f:
        data = pickle.load(f)
    n_classes = 4
    print("Done loading data!\n")
    X_train, X_test, y_train, y_test = data['train']["x"], data["val"]["x"], data["train"]["y"], data["val"]['y']
    data_dim = len(X_train[0])
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.9)
    # X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3)
    print("Number of classes: %d" % n_classes)
    print("Data dimension: %d" % data_dim)
    print("Number of training samples: %d" % (len(y_train)))
    print("Number of testing samples: %d\n" % (len(y_test)))

    # predict with sklearn svm

    clf = RandomForestClassifier()
    start_knn = time.time()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    t = (time.time() - start_knn)/len(y_test)
    acc = accuracy_score(y_test, predict)
    print("Predict time for random forest: %.4f\n" % t)
    print("Accuracy for random forest: %.4f\n" % acc)

    print("-----------------------------------------------\n")
    return predict, y_test


def evaluate(model_name, y_actual, y_pred, class_names):
    print(
        f"model {model_name}: \n"
        f" {classification_report(y_true=y_actual, y_pred=y_pred, target_names=class_names, digits = 4)}")
    make_confusion_matrix_true(y_true=y_actual, y_pred=y_pred, class_names=class_names, save_as_name=model_name)


def main():
    '''
    Test CDNN function
    '''
    for i in range(1, 21):
        k = i
        # k = (i+1)*4
        print("Testing with k = %d\n" % (k))
        test_candy(k)
    # 	# test_iris(k)
    # 	# test_digits(k)
    # 	# test_breast_cancer(k)
    # class_names = ["broken", "holey", "good", "smaller"]
    # predict_dict, y_test_dict = test_candy(k=4)
    # for model_name, value in predict_dict.items():
    #     evaluate(model_name, y_actual=y_test_dict[model_name], y_pred=value, class_names=class_names)
    # y_test, predict = test_candy_svm()
    # evaluate("SVM", y_actual=y_test, y_pred=predict, class_names=class_names)
    # y_test, predict = test_candy_random_forest()
    # evaluate("RandomForest", y_actual=y_test, y_pred=predict, class_names=class_names)


if __name__ == '__main__':
    main()
