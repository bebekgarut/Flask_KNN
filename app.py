from flask import Flask, render_template, request
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

app = Flask(__name__)

# fungsi hitung knn
def knn_classification(train_data, test_data, k):
    
    X_train = train_data.iloc[:, :-1]  # Atribut Training
    y_train = train_data.iloc[:, -1]   # Label Traning

    X_test = test_data.iloc[:, :-1]  # Atribut Testing
    y_test = test_data.iloc[:, -1]   # Label Testing

    # Konversi ke numeric
    label_encoders = {}
    for column in X_train.columns:
        if X_train[column].dtype == 'object':  #
            le = LabelEncoder()
            X_train[column] = le.fit_transform(X_train[column])
            X_test[column] = le.transform(X_test[column])
            label_encoders[column] = le

    # training model
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prediksi
    predictions = knn.predict(X_test)

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
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
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

if __name__ == '__main__':
    app.run(port=5000, debug=True)
