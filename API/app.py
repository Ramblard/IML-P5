
import flask
from flask import Flask, render_template, request, jsonify, redirect
import sys
sys.path.append('./scripts/')
from predictions import *


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        return redirect('/tags?text=' + text)
    return render_template('index.html')


@app.route('/tags', methods=['GET', 'POST'])
def tags():
    if request.method == 'POST':
        text = request.form['text']
        predicted_tags = predire(text)
        nombre_tags = len(predicted_tags)
        predicted_tags = ' '.join(predicted_tags)

        return jsonify({"question":text, "nombre_tags":nombre_tags, "tags":predicted_tags})

    # Otherwise, handle the GET request
    text = request.args.get('text')
    predicted_tags = predire(text)
    nombre_tags = len(predicted_tags)
    predicted_tags = ' '.join(predicted_tags)

    return jsonify({"question":text, "nombre_tags":nombre_tags, "tags":predicted_tags})


if __name__ == "__main__":
    app.run(debug=True)