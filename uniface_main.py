from flask import Flask, render_template, url_for, request, jsonify
import glob
import os
import cv2 as cv
import models.LSTM as LSTM
import models.CNN as CNN
import models.ResNet as ResNet
import models.NN as NN
app = Flask(__name__)

models = ["LSTM", "CNN", "NN", "ResNet"]


def scan_images(path="image_db/"):
    images = {}
    found_files = glob.glob("static/" + path + "*")
    if not len(found_files):
        return "No file structure found"
    for file in found_files:
        images[file] = []
        for image in glob.glob(file + "/*.jpg"):
            images[file].append(image)
    return images


def list_dbs():
    dbs = {}
    found_dbs = [dir for dir in os.listdir("./static/") if os.path.isdir(os.path.join("./static/", dir))]
    for db in found_dbs:
        dbs[db] = len(glob.glob("static/" + db + "/*"))
    return dbs


@app.route("/")
@app.route("/home", methods=["GET"])
def home():
    if request.method == "GET":
        pass
    return render_template("home.html", models=models, dbs=list_dbs(), faces=scan_images())


@app.route("/add")
def add_new_dataset():
    return render_template("add.html")


@app.route("/API/CNN", methods=["POST"])
def API_CNN():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_to_predict= request.json["image2predict"]
    idx = img_to_predict.index("static")
    img_to_predict.replace(img_to_predict[:23], "")
    print(img_to_predict)
    CNN.CNN(img_to_predict)
    return jsonify(200, "success")


@app.route("/API/NN", methods=["POST"])
def API_NN():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_to_predict= request.json["image2predict"]
    idx = img_to_predict.index("static")
    img_to_predict.replace(img_to_predict[:23], "")
    print(img_to_predict)
    NN.NN(img_to_predict)
    return jsonify(200, "success")


@app.route("/API/LSTM", methods=["POST"])
def API_LSTM():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_to_predict= request.json["image2predict"]
    idx = img_to_predict.index("static")
    img_to_predict.replace(img_to_predict[:23], "")
    print(img_to_predict)
    LSTM.LSTM(img_to_predict)
    return jsonify(200, "success")


@app.route("/API/ResNet", methods=["POST"])
def API_ResNet():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_to_predict= request.json["image2predict"]
    idx = img_to_predict.index("static")
    img_to_predict.replace(img_to_predict[:23], "")
    print(img_to_predict)
    ResNet.ResNet(img_to_predict)
    return jsonify(200, "success")


@app.route("/API/images", methods=["GET"])
def get_images():
    imgs = request.args.get("imgs", "", type=str)
    print(imgs)
    return jsonify(imgs)


@app.route("/API/db", methods=["POST"])
def API_DB():
    if not request.method == "POST":
        return(400, "Wrong request")
    opt = request.json["opt"] + "/"
    print("opt value", opt)
    return jsonify(scan_images(opt))


@app.route("/API/test", methods=["POST"])
def API_test():
    imgs = request.json["data"]
    print(imgs)
    return imgs


@app.route("/API/group", methods=["POST"])
def API_group():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_locations = []
    group = request.json["group"]
    group = group.replace("static/", "")
    first_dir = group.find("/")
    db = group[:first_dir]
    group = group[first_dir+1:]
    second_dir = group.find("/")
    group = group[:second_dir]
    found_dbs = [dir for dir in os.listdir("./static/") if os.path.isdir(os.path.join("./static/", dir))]
    if db in found_dbs:
        if "static/" + db + "/" + group in glob.glob("static/" + db + "/*"):
            for img in glob.glob("./static/" + db + "/" + group + "/*.jpg"):
                img_locations.append(img)
            #return {"data": img_locations}
            return jsonify(img_locations)
        else: # TODO add try catch after testing
            return jsonify({"error": "Failed to find selected group"})
    else:
        return jsonify({"error": "Failed to find selected database"})


def load_imgs_hardcoded():
    images = []
    targets = []
    num = 0
    imgs = glob.glob("../datasets/*.jpg")
    for img in imgs:
        images.append(cv.imread(img))
        img = img.replace("../datasets/", "")
        img = img.replace(".jpg", "")
        if img[-2:].isdigit(): num = img[-2:]
        elif img[-1:].isdigit(): num = img[-1:]
        img = img.replace("_", "")
        targets.append(num)

    return images, targets


def find_images():
    groups = {}
    images = []
    imgs = glob.glob("../tf_datasets/**/*.jpg")
    for img in imgs:
        img = img.replace(".jpg", "")
        x = special_chars(img, "/")
        for ii in x:
            if ii == 0:
                parts = img.partition("/")
            else:
                parts = parts[2].partition("/")
        images.append(parts.partition("_")[2])

    for img in imgs:
        images.append(img)
    for img in images:
        img = img.replace(".jpg", "")
        x = img.partition("/")
        # TODO: check how many "/" are in the directory
        if "/" in x[2]:
            x = x[2].partition("/")
        x = x[2].partition("_")
        groups[x[-2:]].append(x[0])


def special_chars(string, char):
    char_count = 0
    for i in string:
        if i == char:
            char_count += 1
    return char_count


if __name__ == "__main__":
    app.run(debug=True)
