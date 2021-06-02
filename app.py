from flask import Flask, request, jsonify
from PIL import Image
import os
import datetime
from shubh_predict import predict


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return jsonify({"202" : "try our post requests"})

@app.route("/upload", methods=["POST"])
def process_image():
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file)

    print(request.files)

    target = os.path.join(APP_ROOT, 'images/')

    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    
    file_name = file.filename

    print("{} is the file name".format(file_name))

    time = str(datetime.datetime.today().strftime('%H-%M-%S'))

    date = str(datetime.date.today())

    extension = os.path.splitext(file_name)[1]

    new_file_name = time + "_" + date + extension

    destination = "/".join([target,new_file_name])

    print("Acceptincoming file:",file_name)

    print("new file name:",new_file_name)

    print("save it ot:",destination)

    resized_im = img.resize((224,224))

    resized_im.save(destination)

    ans = predict(destination)

    return jsonify({'msg': 'success',"ans" : ans, 'size': [resized_im.width, resized_im.height]})

if __name__ == "__main__":
    app.run(port=4555, debug=True)