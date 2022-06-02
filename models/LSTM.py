import numpy as np
import tensorflow as tf
from keras import models, layers, metrics, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import datetime
import seaborn as sn
from keras.preprocessing.image import ImageDataGenerator
import time


def gather_images(dir):
    train = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True)
    test = ImageDataGenerator()
    train_set = train.flow_from_directory(
        dir, target_size=(64, 64), batch_size=32, class_mode="categorical", color_mode="grayscale"
    )
    test_set = test.flow_from_directory(
        dir, target_size=(64, 64), batch_size=32, class_mode="categorical", color_mode="grayscale"
    )
    faces = {}
    for faceValue, faceName in zip(train_set.class_indices.values(),
            train_set.class_indices.keys()):
        faces[faceValue] = faceName
    test_labels = test_set.classes
    return train_set, test_set, test_labels, faces


def LSTM_build(img_predict="../static/image_db/4/34_4.jpg", dir="../static/image_db/"):
    debug = False
    train_set, test_set, test_labels, faces = gather_images(dir)
    input_shape = (64, 64)

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:, :, :, 0], input_shape=(*input_shape, 1)))
    model.add(layers.LSTM(64, activation="tanh", recurrent_activation="sigmoid", kernel_initializer="glorot_normal",
        kernel_regularizer=None, return_sequences=True, recurrent_dropout=0.2))#, activation="sigmoid"))
    model.add(layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2))#, activation="sigmoid"))
# model.add(layers.LSTM(64))
#    model.add(layers.Dropout(0.3))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(len(faces), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
            metrics=["accuracy", metrics.FalseNegatives(), metrics.FalsePositives(),
            metrics.TruePositives(), metrics.TrueNegatives()])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    fit_results = model.fit(train_set, epochs=30, validation_data=test_set, callbacks=[tensorboard_callback] )
    preds = model.predict(test_set)
    eval_score = model.evaluate(test_set)
    model.save('./LSTM_model.h5')

    img_path = img_predict
    predict_image = preprocessing.image.load_img(img_path, target_size=(64, 64))
    predict_image = preprocessing.image.img_to_array(predict_image)
    predict_image = np.expand_dims(predict_image, axis=0)
    result = model.predict(predict_image)
    print("Predicted face:", result)


    plt.style.use("ggplot")
    plt.plot(np.arange(0, 30), fit_results.history["false_negatives"], label="False Negatives")
    plt.plot(np.arange(0, 30), fit_results.history["false_positives"], label="False Positives")
#plt.plot(np.arange(0, 30), fit_results.history["true_negatives"], label="True Negatives")
    plt.plot(np.arange(0, 30), fit_results.history["true_positives"], label="True Positives")
    plt.title("Face classification")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend(loc="lower left")
    plt.figure()

    plt.title("Accuracy and training loss")
    plt.plot(np.arange(0, 30), fit_results.history["accuracy"], label="Accuracy")
    plt.plot(np.arange(0, 30), fit_results.history["loss"], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Metrics")
    plt.legend(loc="lower left")
    #plt.show()

    import sys
    if debug:
        with open("LSTM_test17.txt", "a+") as file:
            sys.stdout = file
            print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Evaluation scores", model.metrics_names, eval_score)
            print(fit_results.history["loss"])
            print(fit_results.history["accuracy"])
            print(fit_results.history["false_negatives"])
            print(fit_results.history["false_positives"])
            print(fit_results.history["true_negatives"])
            print(fit_results.history["true_positives"])
            model.summary()

            file.close()

def LSTM(img_predict=["../static/image_db/4/31_4.jpg"], dir="../static/image_db/"):
    train_set, test_set, test_labels, faces = gather_images(dir)
    try:
        new_model = tf.keras.models.load_model('./LSTM_model.h5')
    except IOError:
        return("Pre-trained model not found")
    eval_score = new_model.evaluate(test_set)

    test_predictions = new_model.predict(test_set, verbose=1)
    y_pred = np.argmax(test_predictions, axis=-1)

    matrix = confusion_matrix(test_labels, y_pred)
    matrix = sn.heatmap(matrix, annot=True)
    plt.show()
    img_path = img_predict
    try:
        for img_path in img_predict:
            predict_image = preprocessing.image.load_img(img_path, target_size=(64, 64))
            predict_image = preprocessing.image.img_to_array(predict_image)

            predict_image = np.expand_dims(predict_image, axis=0)
            result = new_model.predict(predict_image)
            print("Predicted face:", result)
    except:
        return "Could not find the image"

if __name__ == "__main__":
    start_time = time.monotonic()
    LSTM()
    end_time = time.monotonic()
    print(datetime.timedelta(seconds=end_time - start_time))
