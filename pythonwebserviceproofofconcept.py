from flask import *

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    output = ""
    if request.method == "POST":
        input_string = request.form.get("input_string")
        # Perform any processing on the input string here
        # For simplicity, we'll just reverse it
        output = input_string[::-1]

    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(debug=True)
