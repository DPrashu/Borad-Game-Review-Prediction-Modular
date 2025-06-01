from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,predict_pipe

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def get_results():
    type = float(request.form['type'])
    yearpublished = int(request.form['yearpublished'])
    minplayers = int(request.form['minplayers'])
    maxplayers = int(request.form['maxplayers'])
    playingtime = int(request.form['playingtime'])
    minage = int(request.form['minage'])
    users_rated = int(request.form['users_rated'])
    bayes_average_rating = float(request.form['bayes_average_rating'])
    total_traders = int(request.form['total_traders'])
    total_wanters = int(request.form['total_wanters'])
    total_wishers = int(request.form['total_wishers'])
    average_weight = float(request.form['average_weight'])

    obj = CustomData(type,yearpublished,minplayers,maxplayers,playingtime,minage,users_rated,bayes_average_rating,total_traders,
            total_wanters,total_wishers,average_weight)

    data = obj.get_data_as_DataFrame()
    obj2 = predict_pipe()

    result = obj2.predict(data)

    return render_template('results.html',prediction=result)

if (__name__ == '__main__'):
    app.run(host='127.0.0.1', port=5000, debug=True)