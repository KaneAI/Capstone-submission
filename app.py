from flask import Flask, render_template, request
from markupsafe import Markup
import model

app = Flask(__name__)

@app.route("/", methods=['POST','GET'])
def home():
    if request.method=='POST':
         # Accessing form data
        username = request.form['username']
        results = model.get_recommendations(username)
        return render_template("index.html", html_results=Markup(results.to_html(header="true", table_id="table")))
    if request.method=='GET':
        return render_template("index.html", html_results="")
    return render_template("index.html")

@app.route("/submit")
def submit():
    return "Hello from submit page"

if __name__ == '__main__':
    app.run(debug=True)
