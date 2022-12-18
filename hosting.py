import pickle
import tensorflow as tf
import numpy as np
from tensorflow import keras

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

model = tf.keras.models.load_model("soil_classifier.h5")
class_ = ['clay', 'gravel', 'humus', 'sand', 'silt']
image = tf.keras.preprocessing.image

ans =classify("istockphoto-500613911-170667a.jpg")

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

print({
    "potassium ": a,
    "nitrogen": b,
    "phosphorus": c
})

