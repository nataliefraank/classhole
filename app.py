from flask import Flask, render_template, request

from flask import Flask, jsonify, request

import requests
import json
import time

#!/usr/bin/env python
# coding: utf-8

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers

import string
import re


batch_size = 32
raw_train_ds = keras.utils.text_dataset_from_directory(
    "data_formatted/unbalanced/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, f"[{re.escape(string.punctuation)}]", ""
    )

max_features = 2000
embedding_dim = 64
sequence_length = 250

vectorize_layer = keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)

text_ds = raw_train_ds.map(lambda x, y: x)


vectorize_layer.adapt(text_ds)


def manual_predict(man_text):
    vec_text = vectorize_layer(tf.constant([man_text]))
    return model.predict(vec_text)

def manual_odds(man_text):
    result = manual_predict(man_text)
    return result.tolist()[0][0]

# to be filled in with our appropriate labels
def manual_bin(man_text):
    if manual_odds(man_text) >= 0.5:
        return 'positive'
    else:
        return 'negative'


model = keras.models.load_model('main-model.keras')

# begin app definition
app = Flask(__name__)

# no modification required beyond function name
@app.route('/predict')
def predict():

    query = request.args['q']

    prediction_odds = manual_odds(query)
    if prediction_odds >= 0.5:
        label = 'The A*hole'
    else:
        label = 'Not the A*hole'

    result = {
        'raw_odds': prediction_odds,
        'label': label

    }

    response = jsonify(result)
    
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == '__main__':
    app.run(debug=False)