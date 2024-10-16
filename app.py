from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# fungsi hitung knn
def knn_classification(train_data, test_data, k):
    
    X_train = train_data.iloc[:, :-1]  # Atribut Training
    y_train = train_data.iloc[:, -1]   # Label Traning

    X_test = test_data.iloc[:, :-1]  # Atribut Testing
    y_test = test_data.iloc[:, -1]   # Label Testing

      # Cek tipe data dan missing values
    print("Train Data Types:")
    print(X_train.dtypes)
    print("Test Data Types:")
    print(X_test.dtypes)

    print("Missing values in Train Data:")
    print(X_train.isnull().sum())
    print("Missing values in Test Data:")
    print(X_test.isnull().sum())

    # Isi missing values jika ada
    X_train.fillna(method='ffill', inplace=True)
    X_test.fillna(method='ffill', inplace=True)

    print("Train Data Columns:")
    print(X_train.columns)
    print("Test Data Columns:")
    print(X_test.columns)

    # Konversi ke OneHotEncoder (hanya untuk kolom non-numeric)
    onehot_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Mengaplikasikan OneHotEncoder hanya pada kolom non-numeric
    X_train_encoded = onehot_enc.fit_transform(X_train)
    X_test_encoded = onehot_enc.transform(X_test)

    # Training model
    knn = KNeighborsClassifier(n_neighbors=k, metric='hamming')
    knn.fit(X_train_encoded, y_train)

    # Prediksi
    predictions = knn.predict(X_test_encoded)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)

    # Return prediksi dan label testing
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

@app.route("/option")
def option():
    return render_template('rekomen.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)
