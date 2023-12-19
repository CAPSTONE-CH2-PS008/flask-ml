import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from keras.models import load_model

app = Flask(__name__)

model = load_model('../app/models/best_classification_model.h5', compile=False)

@app.route("/api/predict", methods=["post"])
def predict():
    try:
        data = request.get_json()
        input_data = {
            "Bobot" : tf.constant([[data["Bobot"]]], dtype=tf.float32),
            "Activity" : tf.constant([[data["Activity"]]], dtype=tf.float32),
            "Start_Time" : tf.constant([[data["Start_Time"]]], dtype=tf.float32),
            "End_Time" : tf.constant([[data["End_Time"]]], dtype=tf.float32),
            "Interest" : tf.constant([[data["Interest"]]], dtype=tf.float32)
        }
        result = model.predict(input_data)
        result = int(np.argmax(result))
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)