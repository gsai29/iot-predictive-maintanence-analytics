from flask import Flask, request, jsonify, render_template, send_from_directory
import numpy as np
import tensorflow as tf
from flask_cors import CORS
tf.compat.v1.disable_eager_execution()
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
app = Flask(__name__)
CORS(app)

model_path = 'C:/Users/vigne/OneDrive/Documents/ML/Predictive-Maintenance-master/Predictive-Maintenance/saved_model/my_model'
sess = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph(f'{model_path}.meta')
saver.restore(sess, model_path)

graph = tf.compat.v1.get_default_graph()
X = graph.get_tensor_by_name("X:0")
f_outputs = graph.get_tensor_by_name("MatMul:0")
data_buffer = []

def loadCSV():
    folder_path = 'C:/Users/vigne/OneDrive/Documents/ML/Predictive-Maintenance-master/Predictive-Maintenance/dataset/raw_data'
    file_name = 'part-00000-d6f2018b-b707-4246-82b2-a60e146f330c-c000.csv'
    full_path = f'{folder_path}/{file_name}'
    df = pd.read_csv(full_path).sort_values('TimeStamp', ascending=True).reset_index(drop=True)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
    columns_to_drop = [
        '::[scararobot]Ax_J1.PositionCommand', '::[scararobot]Ax_J1.TorqueFeedback',
        '::[scararobot]Ax_J2.PositionCommand', '::[scararobot]Ax_J2.TorqueFeedback',
        '::[scararobot]Ax_J3.TorqueFeedback', '::[scararobot]Ax_J6.TorqueFeedback',
        '::[scararobot]ScanTimeAverage', '::[scararobot]Ax_J6.PositionCommand',
        '::[scararobot]Ax_J3.PositionCommand'
    ]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
    df['Total'] = df.select_dtypes(include=['float64', 'float32']).sum(axis=1)
    return df['Total']

def anomalyDetection(data, outliers_fraction=0.01):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    model = OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.01)
    model.fit(data_scaled)
    preds = model.predict(data_scaled)
    anomalies = np.where(preds == -1)[0]  # Return indices of anomalies
    return anomalies

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/dashboard')
def serve_dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    global data_buffer

    input_data = request.json['input']
 
    data_buffer.extend(input_data)

    if len(data_buffer) >= 100:

        input_data = data_buffer[-100:]

        input_data = np.array(input_data).reshape(1, 100, 1)

        predictions = sess.run(f_outputs, feed_dict={X: input_data})

        data_buffer = data_buffer[-100:]

        return jsonify({'predictions': predictions.tolist()})
    else:
        return jsonify({'status': 'waiting for more data', 'current_length': len(data_buffer)})

@app.route('/heartbeat', methods=['GET'])
def heartbeat():
    status = 'Normal' if random.random() > 0.5 else 'Warning'
    return jsonify({'heartbeat': status})

@app.route('/detect_anomalies', methods=['POST'])
def detect_anomalies():
    predictions = np.array(request.json['predictions'])
    anomalies = anomalyDetection(predictions.flatten(), 0.01)
    return jsonify({'anomalies': anomalies.tolist()})


@app.route('/<path:path>')
def static_proxy(path):
    # Route for serving the React app's static files
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(debug=True)
