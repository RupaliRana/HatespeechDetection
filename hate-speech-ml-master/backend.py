from flask import Flask, request, render_template,jsonify
import numpy as np
import pickle

app = Flask(__name__)

mlmodel = pickle.load(open("classifier_file.pkl","rb"))
cv = pickle.load(open("cv.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/detect",methods = ["POST"])

def detect():
    if request.method == 'POST':
        Input = request.form['Input']
        df=cv.transform([Input]).toarray()
        output=mlmodel.predict(df)

    return render_template("index.html", Output = "{}".format(output))

if __name__ == "__main__":
    app.run(debug=True)