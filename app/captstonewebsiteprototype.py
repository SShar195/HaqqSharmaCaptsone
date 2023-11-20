from flask import *

app = Flask(__name__)

@app.route("/")
def redirect_to_haar():
    return redirect("/haar")


@app.route('/haar')
def haar():
    # You can read the code snippet from a file (e.g., code.txt)
    # and the image from a file (e.g., image.jpg) and pass them to the template.
    with open("app\\haar.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'haar.html', code=code_snippet, image='Haar Theoretical Results')

@app.route("/twodesign")
def twoDesign():
    # You can read the code snippet from a file (e.g., code.txt)
    # and the image from a file (e.g., image.jpg) and pass them to the template.
    with open("app\\two_design.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'twodesign.html', code=code_snippet, image='Two-Design Theoretical Results.png')

@app.route("/clifford")
def clifford():
    # You can read the code snippet from a file (e.g., code.txt)
    # and the image from a file (e.g., image.jpg) and pass them to the template.
    with open("app\\clifford.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'clifford.html', code=code_snippet, image='Two-Design Theoretical Results.png')
    

if __name__ == "__main__":
    app.run(debug=True)
