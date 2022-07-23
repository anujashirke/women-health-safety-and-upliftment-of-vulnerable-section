

from flask import Flask, render_template, request,redirect,url_for
import numpy as np
import pickle


app = Flask(__name__)
model = pickle.load(open('Stroke.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('Home.html')

@app.route('/stroke',methods=['GET'])
def Stroke():
    return render_template('stroke.html')

@app.route('/explore',methods=['GET'])
def explore():
    return render_template('explore.html')
    
@app.route('/result',methods=['GET'])
def result():
    return render_template('result.html')

@app.route('/result1',methods=['GET'])
def result1():
    return render_template('result1.html')

@app.route('/yoga',methods=['GET'])
def yoga():
    return render_template('yoga.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        ever_married = int(request.form['ever_married'])
        Residence_type = int(request.form['Residence_type'])


        work_type = request.form['work_type']

        if work_type == 'Never_worked':
            work_type_Never_worked = 1
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 0

        if work_type == 'Private':
            work_type_Never_worked = 0
            work_type_Private = 1
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 0

        elif work_type == "Self_employed":
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 1
            work_type_children = 0
            work_type_Govt_job = 0

        elif work_type == "children":
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 1
            work_type_Govt_job = 0

        else:
            work_type_Never_worked = 0
            work_type_Private = 0
            work_type_Self_employed = 0
            work_type_children = 0
            work_type_Govt_job = 1


        smoking_status = request.form['smoking_status']

        if smoking_status == "formerly_smoked":
            smoking_status_formerly_smoked = 1
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 0
            smoking_status_Unknown = 0

        elif smoking_status == "never_smoked":
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 1
            smoking_status_Smokes = 0
            smoking_status_Unknown = 0

        elif smoking_status == "Smokes":
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 1
            smoking_status_Unknown = 0

        else:
            smoking_status_formerly_smoked = 0
            smoking_status_never_smoked = 0
            smoking_status_Smokes = 0
            smoking_status_Unknown = 1

        

        glucose = request.form['avg_glucose_level']

        if glucose == 'gNormal':
            gNormal = 1
            gModerate = 0
            gHigh = 0

        if glucose == 'gModerate':
            gNormal = 0
            gModerate = 1
            gHigh = 0

        elif glucose == "gHigh":
            gNormal = 0
            gModerate = 0
            gHigh = 1

        bmi = request.form['bmi']

        if bmi == 'bNormal':
            bNormal = 1
            bModerate = 0
            bHigh = 0

        if bmi == 'bModerate':
            bNormal = 0
            bModerate = 1
            bHigh = 0

        elif bmi == "bHigh":
            bNormal = 0
            bModerate = 0
            bHigh = 1

        values = np.array([[age, hypertension, heart_disease, ever_married,
                            Residence_type,
                            work_type_Never_worked, work_type_Private,work_type_Self_employed, work_type_children,
                            smoking_status_formerly_smoked, smoking_status_never_smoked, smoking_status_Smokes,gNormal,gModerate,gHigh,bNormal,bModerate,bHigh]])
        prediction = model.predict(values)

        if prediction==1:
            return redirect(url_for('result1'))
        elif prediction==0:
            return redirect(url_for('result'))
    return render_template('result.html', prediction=prediction)

if __name__=='__main__':
    app.run(debug=True,use_reloader=False)


