
import numpy as np
#need to 'conda install flask' for this to work
from flask import Flask, abort, jsonify, request
import pickle


my_random_forest = pickle.load(open("iris_rfc.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
    #all kinds of error checking should go here
    data = request.get_json(force=True)
    #convert our json to a numpy array
    predict_request = [data['sl'],data['sw'],data['pl'], data['pw']] 
    predict_request = np.array(predict_request).reshape(1,-1)
    #np array goes into random forest, prediction comes out
    y_hat = my_random_forest.predict(predict_request)
    #return our prediction
    output = {'y_hat': int(y_hat[0])}
    print(output)
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 9000, debug = True)


