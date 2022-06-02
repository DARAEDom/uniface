from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn import svm
from sklearn.utils.fixes import loguniform
import numpy as np
import cv2 as cv
import glob
from skimage.transform import resize
from skimage.io import imread

class_names = ["a", "b", "c", "d", "e", "f", "g", "h", "k", "l", "m", "n", "o", "p", "q", "u", "v"]

def PCA_calculation(img_size, train, test):

    pca = PCA(n_components=0.85, svd_solver="auto", iterated_power="auto")
    pca_model = pca.fit(train)

    train_pca = pca_model.transform(train)
    test_pca = pca_model.transform(test)

    return train_pca, test_pca


def SVC_rbf(X, y):
    X_images, y_images, X_labels, y_labels = train_test_split(X, y, test_size=0.25)

    X_images, y_images = X_images / 255.0, y_images / 255.0

    scaler = StandardScaler()
    X_images = scaler.fit_transform(X_images)
    y_images = scaler.fit_transform(y_images)
    X_images_pca, y_images_pca = PCA_calculation(80*70, X_images, y_images)

    SVC_rbf = svm.SVC(kernel="rbf", C=10.0, gamma=0.001,
            decision_function_shape="ovo", max_iter=100)

    SVC_rbf_fit = SVC_rbf.fit(X_images_pca, X_labels)

    rbf_predict = SVC_rbf_fit.predict(y_images_pca)

    ConfusionMatrixDisplay(confusion_matrix(y_labels, rbf_predict,)).plot()
    plt.show()

    print("RBF kernel", classification_report(y_labels, rbf_predict,
        target_names=class_names))


def SVC_linearv2(X, y):
    X_images, y_images, X_labels, y_labels = train_test_split(X, y, test_size=0.25)

    X_images, y_images = X_images / 255.0, y_images / 255.0

    scaler = StandardScaler()
    X_images = scaler.fit_transform(X_images)
    y_images = scaler.fit_transform(y_images)
    X_images_pca, y_images_pca = PCA_calculation(80*70, X_images, y_images)

    LinearSVC = svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False,
        C=10.0, multi_class="crammer_singer",
        fit_intercept=True, class_weight=None, max_iter=10)

    LinearSVC.fit(X_images_pca, X_labels)
    LinearSVC_predict = LinearSVC.predict(y_images_pca)
    ConfusionMatrixDisplay(confusion_matrix(y_labels, LinearSVC_predict,)).plot()
    plt.show()
    print("Linear kernel", classification_report(y_labels, LinearSVC_predict,
        target_names=class_names))
    
    #    predict_img = cv.imread(predict_img)
    #    predict_img = cv.cvtColor(predict_img, cv.COLOR_BGR2GRAY)
#   #     predict_img = resize(predict_img, (26, 26))
    #    predict_img = np.asarray(predict_img)
    #    predict_img = predict_img.astype(float)
    #    predict_img = predict_img.reshape(predict_img.shape[0], predict_img.shape[1])
    #    predicton = [predict_img.flatten()]
    #    prob = LinearSVC.predict_proba(predicton)
    #    for key, val in enumerate(class_names):
    #        print("Probability for ", val, "is : ", class_names[LinearSVC.predict(predicton)[0]])


def SVC_poly(X, y):
    X_images, y_images, X_labels, y_labels = train_test_split(X, y, test_size=0.25)

    X_images, y_images = X_images / 255.0, y_images / 255.0

    scaler = StandardScaler()
    X_images = scaler.fit_transform(X_images)
    y_images = scaler.fit_transform(y_images)
    X_images_pca, y_images_pca = PCA_calculation(80*70, X_images, y_images)

    SVC_poly = svm.SVC(kernel="poly", degree=3, gamma="scale", coef0=0.0,
            tol=1e-3, class_weight=None, probability=True,
            decision_function_shape="ovr", max_iter=10)

    SVC_poly.fit(X_images_pca, X_labels)

    poly_predict = SVC_poly.predict(y_images_pca)

    ConfusionMatrixDisplay(confusion_matrix(y_labels, poly_predict,)).plot()
    plt.show()
    print("Linear kernel", classification_report(y_labels, poly_predict,
        target_names=class_names))


def SVC_linear(X, y):
    X_images, y_images, X_labels, y_labels = train_test_split(X, y, test_size=0.25)

    X_images, y_images = X_images / 255.0, y_images / 255.0

    scaler = StandardScaler()
    X_images = scaler.fit_transform(X_images)
    y_images = scaler.fit_transform(y_images)
    X_images_pca, y_images_pca = PCA_calculation(80*70, X_images, y_images)

    SVC_linear = svm.SVC(kernel="linear", C=1.0, tol=1e-3, class_weight=None,
        decision_function_shape="ovr", max_iter=10, probability=True)

    SVC_linear.fit(X_images_pca, X_labels)
    linear_predict = SVC_linear.predict(y_images_pca)

    ConfusionMatrixDisplay(confusion_matrix(y_labels, linear_predict,)).plot()
    plt.show()
    print("Linear kernel", classification_report(y_labels, linear_predict,
        target_names=class_names))

    #img=imread(predict_img)
    #img_resize=resize(img,(28,28))
    #l=[img_resize.flatten()]
    #probability=SVC_linear.predict_proba(l)
    #for ind,val in enumerate(class_names):
    #    print(f'{val} = {probability[0][ind]*100}%')
    #print("The predicted image is : "+class_names[SVC_linear.predict(l)[0]])
    print("Linear kernel", classification_report(y_labels, linear_predict, target_names=class_names))



def data_preprocessing(X, y):
    X = np.asarray(X)
    X = X.astype(float)
    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
    y = np.asarray(y)
    y = y.astype(float)
    y = y.reshape(-1, 1)

    return X, y


def import_imgs():
    images = []
    targets = []
    num = 0
    imgs = glob.glob("../tf_datasets/*.jpg")
    for img in imgs:
        images.append(cv.imread(img))
        img = img.replace("../tf_datasets/", "")
        img = img.replace(".jpg", "")
        if img[-2:].isdigit(): num = img[-2:]
        elif img[-1:].isdigit(): num = img[-1:]
        img = img.replace("_", "")
        targets.append(num)
    return images, targets


if __name__ == "__main__":
    X, y = import_imgs()
    X, y = data_preprocessing(X, y)
#    SVC_rbf(X, y)
#    SVC_linear(X, y)
#    SVC_poly(X, y)
    SVC_linearv2(X, y)
