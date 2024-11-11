from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

knn = None
onehot_enc = None

def knn_classification(train_data, test_data, k):
    global knn, onehot_enc
    
    X_train = train_data.iloc[:, :-1]  # Atribut Training
    y_train = train_data.iloc[:, -1]   # Label Traning

    X_test = test_data.iloc[:, :-1]  # Atribut Testing
    y_test = test_data.iloc[:, -1]   # Label Testing

    onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    X_train_encoded = onehot_enc.fit_transform(X_train)
    X_test_encoded = onehot_enc.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_encoded, y_train)

    predictions = knn.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    return predictions, y_test.values, accuracy, report


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    training_data = None
    testing_data = None

    if request.method == 'POST':
      
        train_file = request.files['train_file']
        test_file = request.files['test_file']
        k_value = int(request.form['k_value'])

        try:
            train_data = pd.read_csv(train_file, header=None)
            test_data = pd.read_csv(test_file, header=None)
        except Exception as e:
            return f"Error reading files: {str(e)}", 400

        predictions, actual_labels, accuracy, report = knn_classification(train_data, test_data, k_value)

        result = {
                'predictions': predictions,
                'actual_labels': actual_labels,
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
        }
        training_data = train_data
        testing_data = test_data

    return render_template('index.html', result=result, training_data=training_data, testing_data = testing_data, zip=zip)

@app.route("/option", methods=['GET', 'POST'])
def option():
    prediction = None
    if request.method == 'POST':
        status_sosial = request.form.get('status_sosial')
        perawatan = request.form.get('perawatan')
        kondisi_keluarga = request.form.get('kondisi_keluarga')
        jumlah_anak = request.form.get('jumlah_anak')
        perumahan = request.form.get('perumahan')
        keuangan = request.form.get('keuangan')
        masalah_sosial = request.form.get('masalah_sosial')
        kesehatan = request.form.get('kesehatan')
        
        # print(f"Status Sosial: {status_sosial}, Perawatan: {perawatan}, Kondisi Keluarga: {kondisi_keluarga}, Jumlah Anak: {jumlah_anak}, Perumahan: {perumahan}, Keuangan: {keuangan}, Masalah Sosial: {masalah_sosial}, Kesehatan: {kesehatan}")

        input_data = pd.DataFrame([[status_sosial, perawatan, kondisi_keluarga, jumlah_anak, 
                                    perumahan, keuangan, masalah_sosial, kesehatan]])

        input_encoded = onehot_enc.transform(input_data)

        # print(f"Input Encoded: {input_encoded}")

        prediction = knn.predict(input_encoded)[0]

        # print(f"Prediction: {prediction}")

    return render_template('rekomen.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
