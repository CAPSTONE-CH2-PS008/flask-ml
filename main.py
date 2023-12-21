import tensorflow as tf
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from keras.models import load_model
import pickle

app = Flask(__name__)

model = load_model('Mult_Label_Model.h5', compile=False)

@app.route("/")
def index():
    return "Hello World"

@app.route("/api/predict", methods=["post"])
def predict():
    try:
        with open('dataset.pkl', 'rb') as file:
            df = pickle.load(file)

        data = request.get_json()
        """ input_data = {
            "Bobot" : tf.constant([[data["Bobot"]]], dtype=tf.float32),
            "Activity" : tf.constant([[data["Activity"]]], dtype=tf.float32),
            "Start_Time" : tf.constant([[data["Start_Time"]]], dtype=tf.float32),
            "End_Time" : tf.constant([[data["End_Time"]]], dtype=tf.float32),
            "Interest" : tf.constant([[data["Interest"]]], dtype=tf.float32)
        }"""

        input_data = np.array([[
            data["Bobot"],
            data["Activity"],
            data["Start_Time"],
            data["End_Time"],
            data["Interest"]            
        ]])
        result = model.predict(input_data)
        result = int(np.argmax(result) + 1)
        if result == 4 or result == 3:
            print("result = 4")
            return jsonify({
                "result": result,
                "end time" : data['End_Time'],
                "start time" : data['Start_Time']
                })
        else:
            all_good = df[df.apply(search_best, data=data, axis=1)]
            all_good = all_good[all_good['Grade'] == 4]
            index = 3
            
           
            if not all_good.empty:
                print('good')
            else:
                while index >= 1:
                    all_good = df[df.apply(search_best, data=data, axis=1)]
                    all_good = all_good[all_good['Grade'] == index]
                    
                    if not all_good.empty:
                        break  #
                    index = index - 1
            df_result_start = all_good.apply(find_closest_value, data=data, start=True, axis=1)
            df_result_end = all_good.apply(find_closest_value, data=data, start=False, axis=1)
            closset_start = df_result_start.loc[df_result_start['target_difference'].idxmin()]
            closset_end = df_result_end.loc[df_result_end['target_difference'].idxmin()]

            """input_data = {
                "Bobot" : tf.constant([[data["Bobot"]]], dtype=tf.float32),
                "Activity" : tf.constant([[data["Activity"]]], dtype=tf.float32),
                "Start_Time" : tf.constant([[closset_start["Start Time"]]], dtype=tf.float32),
                "End_Time" : tf.constant([[closset_end["Start End"]]], dtype=tf.float32),
                "Interest" : tf.constant([[data["Interest"]]], dtype=tf.float32)
            }"""

            input_data = np.array([[
                data["Bobot"],
                data["Activity"],
                closset_start["Start Time"],
                closset_end["End Time"],
                data["Interest"]            
            ]])
            result = model.predict(input_data)
            result = int(np.argmax(result) + 1)
            return jsonify({
                "result": result,
                "start time" : int(closset_start["Start Time"]),
                "end time" : int(closset_end["End Time"])
                })
            
    except Exception as e:
        return jsonify({"error": str(e)})

def search_best(row, data):
    return (
        row['Bobot'] == data['Bobot']
        and row['Activity'] == data['Activity']
        and row['Interest'] == data['Interest']
    )
    
def find_closest_value(row, data, start):
    if start:
        row['target_difference'] = abs(10000 if row["Start Time"] - data["Start_Time"] == 0 else  row["Start Time"] - data["Start_Time"])
    else:
        row['target_difference'] = abs(10000 if row["End Time"] - data["End_Time"]  == 0 else row["End Time"] - data["End_Time"])
    return row


if __name__ == "__main__":
    app.run(debug=True)