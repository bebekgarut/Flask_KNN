from flask import Flask,redirect,url_for,render_template,request
  
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    # if request.method=='POST':
    #     # Handle POST Request here
    #     return render_template('index.html')
    value = "b"
    array = ["laravel", "react", "flask"]
    return render_template('index.html', valu=value, arr=array)

@app.route('/parsing/<int:nilai>')
def parsing(nilai):
    return '{}'.format(nilai)

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)