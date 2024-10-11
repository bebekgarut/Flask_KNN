from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

from sklearn.preprocessing import LabelEncoder

# fungsi hitung knn
def knn_classification(train_file, test_file, k):
    # baco file
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
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

    # training
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Prediksi
    predictions = knn.predict(X_test)

    # Return prediksi dan label testing
    return predictions, y_test.values


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
      
        train_file = request.files['train_file']
        test_file = request.files['test_file']
        k_value = int(request.form['k_value'])

        if train_file and test_file:
            # Save ke uploads
            train_filename = secure_filename(train_file.filename)
            test_filename = secure_filename(test_file.filename)
            train_file.save(os.path.join(app.config['UPLOAD_FOLDER'], train_filename))
            test_file.save(os.path.join(app.config['UPLOAD_FOLDER'], test_filename))

            # Perform KNN classification
            train_path = os.path.join(app.config['UPLOAD_FOLDER'], train_filename)
            test_path = os.path.join(app.config['UPLOAD_FOLDER'], test_filename)

            predictions, actual_labels = knn_classification(train_path, test_path, k_value)

            result = {
                'predictions': predictions,
                'actual_labels': actual_labels
            }

    return render_template('index.html', result=result, zip=zip)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
