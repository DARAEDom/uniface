from keras import preprocessing, models, layers, initializers, metrics
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import datetime
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import seaborn as sn
import time
import sys


def import_imgs(dir):
    train = ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True, rescale=1/255)
    test = ImageDataGenerator()
    train_set = train.flow_from_directory(
        dir, target_size=(64, 64),
        batch_size=32, class_mode="categorical",
        shuffle=True,
    )
    test_set = test.flow_from_directory(
        dir, target_size=(64, 64),
        batch_size=32, class_mode="categorical"
    )
    faces = {}
    for faceValue, faceName in zip(
        train_set.class_indices.values(), train_set.class_indices.keys()
    ):
        faces[faceValue] = faceName

    return train_set, test_set, 0, faces


def conv_block(model, f, filters, s=2):
    f1, f2, f3 = filters
    model_shortcut = model

    # First Layer
    model = layers.Conv2D(f1, (1, 1), strides=(s, s),
            )(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)

    # Second Layer
    model = layers.Conv2D(f2, (f, f), strides=(1, 1), padding="same",
            )(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)

    # Third Layer
    model = layers.Conv2D(f3, (1, 1), strides=(1, 1), padding="valid",
            )(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)

    # Shortcut
    model_shortcut = layers.Conv2D(filters=f3, kernel_size=(1, 1), strides=(s, s),
            padding="valid",)(model_shortcut)
    model_shortcut = layers.BatchNormalization(axis=3)(model_shortcut)

    # Final Step
    model = layers.Add()([model, model_shortcut])
    model = layers.Activation("relu")(model)
    return model


def id_block(model, f, filters):
    f1, f2, f3 = filters

    model_shortcut = model
    model = layers.Conv2D(f1, (1, 1), strides=(1, 1),
            )(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)

    # Second Layer
    model = layers.Conv2D(f2, (f, f), strides=(1, 1), padding="same",
            )(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)

    # Third Layer
    model = layers.Conv2D(f3, (1, 1), strides=(1, 1), padding="valid",
            )(model)
    model = layers.BatchNormalization(axis=3)(model)

    # Final Step
    model = layers.Add()([model, model_shortcut])
    model = layers.Activation("relu")(model)
    return model



def ResNet(model_shape, classes):
    model_input = layers.Input(model_shape)
    model = layers.ZeroPadding2D((3, 3))(model_input)
    model = layers.Conv2D(64, (7, 7), strides=(2, 2),)(model)
    model = layers.BatchNormalization(axis=3)(model)
    model = layers.Activation("relu")(model)
    model = layers.MaxPooling2D((3, 3), strides=(2, 2))(model)

    model = conv_block(model, 3, [128, 128, 512], 1)
    model = id_block(model, 3, [128, 128, 512])
    model = id_block(model, 3, [128, 128, 512])
    model = id_block(model, 3, [128, 128, 512])

    model = conv_block(model, 3, [256, 256, 1024], 2)
    model = id_block(model, 3, [256, 256, 1024])
    model = id_block(model, 3, [256, 256, 1024])
    model = id_block(model, 3, [256, 256, 1024])
    model = id_block(model, 3, [256, 256, 1024])
    model = id_block(model, 3, [256, 256, 1024])

    model = conv_block(model, 3, [512, 512, 2048], 2)
    model = id_block(model, 3, [512, 512, 2048])
    model = id_block(model, 3, [512, 512, 2048])

    model = layers.AveragePooling2D((2, 2))(model)
    # last parts
    model = layers.Flatten()(model)
    model = layers.Dense(classes, activation="softmax", )(model)

    model = models.Model(inputs = model_input, outputs = model, name="ResNet50")
    return model


def ResNet_build(img_predict="../static/image_db/4/34_4.jpg", dir="../static/image_db/"):
    train = preprocessing.image.ImageDataGenerator(shear_range=0.1,
           zoom_range=0.1, horizontal_flip=True, rescale=1/255)
    test = preprocessing.image.ImageDataGenerator()
    train_set = train.flow_from_directory("../static/image_db/",
            target_size=(64, 64), batch_size=32, class_mode="categorical")
    test_set = test.flow_from_directory("../static/image_db/",
            target_size=(64, 64), batch_size=32, class_mode="categorical")

    faces = {}
    for faceValue, faceName in zip(
            train_set.class_indices.values(), train_set.class_indices.keys()):
        faces[faceValue] = faceName
    model = ResNet((64, 64, 3), len(train_set.class_indices.values()))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
        metrics=["accuracy", metrics.FalseNegatives(), metrics.FalsePositives(),
        metrics.TruePositives(), metrics.TrueNegatives()])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    fit_results = model.fit(train_set, epochs=30, validation_data=test_set, callbacks=[tensorboard_callback])

    eval_score = model.evaluate(test_set)
    model.save('./ResNet.h5')


def ResNet(img_predict=["../static/image_db/4/34_4.jpg"], dir="../static/image_db/"):

    train_set, test_set, test_labels, faces = import_imgs(dir)
    try:
        new_model = tf.keras.models.load_model('./ResNet_model.h5')
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
    debug = False
    train = preprocessing.image.ImageDataGenerator(shear_range=0.1,
           zoom_range=0.1, horizontal_flip=True, rescale=1/255)
    test = preprocessing.image.ImageDataGenerator()
    train_set = train.flow_from_directory("../static/image_db/",
            target_size=(64, 64), batch_size=32, class_mode="categorical")
    test_set = test.flow_from_directory("../static/image_db/",
            target_size=(64, 64), batch_size=32, class_mode="categorical")

    faces = {}
    for faceValue, faceName in zip(
            train_set.class_indices.values(), train_set.class_indices.keys()):
        faces[faceValue] = faceName
    model = ResNet((64, 64, 3), len(train_set.class_indices.values()))
    model.compile(loss="categorical_crossentropy", optimizer="adam",
        metrics=["accuracy", metrics.FalseNegatives(), metrics.FalsePositives(),
        metrics.TruePositives(), metrics.TrueNegatives()])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    fit_results = model.fit(train_set, epochs=30, validation_data=test_set, callbacks=[tensorboard_callback])

    eval_score = model.evaluate(test_set)
    model.save('./ResNet.h5')

    import sys
    if debug:
        with open("ResNet_test.txt", "a+") as file:
            plt.style.use("ggplot")

            plt.plot(np.arange(0, 30), fit_results.history["false_negatives"], label="False Negatives")
            plt.plot(np.arange(0, 30), fit_results.history["false_positives"], label="False Positives")
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
            plt.show()
            sys.stdout = file
            print("Time:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print("Evaluation scores", model.metrics_names, eval_score)
            fit_results.history["loss"]
            fit_results.history["accuracy"]
            fit_results.history["false_negatives"]
            fit_results.history["false_positives"]
            fit_results.history["true_negatives"]
            fit_results.history["true_positives"]
            model.summary()

            file.close()
