import requests, os, uuid, json
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, redirect, url_for, request, render_template, session

import model 
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



@app.route('/', methods=['POST'])
def index_post():
    # Read the values from the form
    original_text = request.form['text']
    print(original_text)

    # Execute the classification process with the KCNN model
    y = model.get_prediction(original_text)

    # Call render template, passing the output value,
    return render_template("index.html", output = y[0][0], text = original_text)
    #return render_template(
    #    'results.html',
    #    result=y,
    #    original_text=original_text,
    #)