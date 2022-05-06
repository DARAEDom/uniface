from flask import Flask, render_template, url_for, request, jsonify
import glob
import os
import cv2 as cv
import models.LSTM as LSTM
import models.ResNet as ResNet
import models.SVC as SVC
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


@app.route("/API/predict", methods=["POST"])
def API_predict():
    if not request.method == "POST":
        return(400, "Wrong request")
    img_to_predict = request.json("image2predict")
    model = request.json("learn_model")
    print(model)
    print(img_to_predict)
    return jsonify("Yes it works")


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
