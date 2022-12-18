from flask import Flask,request,jsonify, json
import werkzeug
import cv2
import tensorflow as tf
import numpy as np
import os
import pickle

class_ = ['clay', 'gravel', 'humus', 'sand', 'silt']
image = tf.keras.preprocessing.image
model = tf.keras.models.load_model("soil_classifier.h5")
def classify(image_fp):
    img = image.load_img(image_fp, target_size = (256, 256))
    img = image.img_to_array(img)

    image_array = img / 255. # scale the image
    img_batch = np.expand_dims(image_array, axis = 0)

    predicted_value = model.predict(img_batch)

    out  = {
      "otherClasses":[
        { "name": "clayey", "value": f"{predicted_value[0][0]}" },
        { "name": "gravel" ,"value": f"{predicted_value[0][1]}" },
        { "name": "humus" , "value": f"{predicted_value[0][2]}" },
        { "name": "sandy" , "value": f"{predicted_value[0][3]}" },
        { "name": "silty" , "value": f"{predicted_value[0][4]}" },
      ],
      "Result": f"{(predicted_value[0][predicted_value.argmax()] * 100):.2f}% {class_[predicted_value.argmax()]}",
      "accuracy_score": f"{predicted_value[0][predicted_value.argmax()]}",
      "accuracy_score_rounded": f"{(predicted_value[0][predicted_value.argmax()] * 100):.2f}",
      "soil_type": f"{class_[predicted_value.argmax()]}"
    }
    return out


app= Flask(__name__)    

@app.route('/npk',methods=["PUT"])
def npk():
    if(request.method == "PUT"):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./uploaded_images/"+filename)
        
        ans = classify("uploaded_images/"+filename)
        sand = float(ans['otherClasses'][3]['value'])
        clay = float(ans['otherClasses'][0]['value'])
        silt = float(ans['otherClasses'][4]['value'])

        sandper = sand/(sand+clay+silt)
        clayper = clay/(sand+clay+silt)
        siltper = silt/(sand+clay+ silt)

        lock = "kpred.pkl"
        locn = "npred.pkl"
        locp = "ppred.pkl"

        model_k = pickle.load(open(lock, 'rb'))
        model_n = pickle.load(open(locn, 'rb'))
        model_p = pickle.load(open(locp, "rb"))

        tst = [sand, clay,silt]

        a= model_k.predict([tst])
        b= model_n.predict([tst])
        c=model_p.predict([tst])

        final_ans = {
                "potassium ": a[0],
                "nitrogen": b[0],
                "phosphorus": c[0]
        }
        response = app.response_class(response=json.dumps(final_ans),
                                  status=200,
                                  mimetype='application/json')
        return response


@app.route('/test', methods=["get"])
def test():
    return jsonify({
        "message": "hello world"
    })


if __name__=="__main__":
    app.run(debug=True, port=5000)

    




