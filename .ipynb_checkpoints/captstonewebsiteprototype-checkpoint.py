from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    # You can read the code snippet from a file (e.g., code.txt)
    # and the image from a file (e.g., image.jpg) and pass them to the template.
    with open("code.txt", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template("index.html", code=code_snippet)

if __name__ == "__main__":
    app.run(debug=True)
