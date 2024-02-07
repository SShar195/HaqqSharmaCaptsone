from flask import *

app = Flask(__name__)

@app.route("/")
def redirect_to_haar():
    return redirect("/haar")


@app.route('/haar')
def haar():
    with open("app/static/haar.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'algorithmTemplate.html', code=code_snippet, image='Haar Theoretical Results.png', name='Haar')

@app.route("/twodesign")
def twoDesign():
    with open("app/static/two_design.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'algorithmTemplate.html', code=code_snippet, image='Two-Design Theoretical Results.png', name="Two Design")

@app.route("/clifford")
def clifford():
    with open("app/static/clifford.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'algorithmTemplate.html', code=code_snippet, image='Two-Design Theoretical Results.png', name='Clifford')
    

@app.route("/pooledpauli")
def pooledpauli():
    with open("app/static/Pooled Pauli.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'algorithmTemplate.html', code=code_snippet, image='Pooled Pauli.png', name='Pooled Pauli')

@app.route("/pooledpaulipreselect")
def pooledpaulipreselect():
    with open("app/static/Pooled Pauli Pre Selection.py", "r") as code_file:
        code_snippet = code_file.read()
    
    return render_template(r'algorithmTemplate.html', code=code_snippet, image='Pooled Pauli Pre-Selection.png', name='Pooled Pauli')

if __name__ == "__main__":
    app.run(debug=True)
