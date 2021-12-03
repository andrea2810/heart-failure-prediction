# -*- coding: utf-8 -*-
# Coded By: Andy Rojas

from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def parse_sex(sex):
    if sex == 'M':
        return 0
    else:
        return 1

def parse_chest_pain(chest_pain):
    # Typical Anigna
    if chest_pain == 'TA':
        return 0
    # Atypical Anigna
    elif chest_pain == 'ATA':
        return 1
    # Non-Anginal Pain
    elif chest_pain == 'NAP':
        return 2
    # Asymptomatic
    else:
        return 3

def parse_restingECG(RestingECG):
    # Normal
    if RestingECG == 'Normal':
        return 0
    # ST-T wave abnormality
    elif RestingECG == 'ST':
        return 1
    # LVH probable or definite left ventricular hypertrophy
    else:
        return 2

def parse_exerciseAngina(ExerciseAngina):
    if ExerciseAngina == 'No':
        return 0
    else:
        return 1

def parse_stSlope(ST_Slope):
    if ST_Slope == 'Up':
        return 0
    elif ST_Slope == 'Flat':
        return 1
    else:
        return 2

def main():

    # Algorithms for classification
    classifiers = {
        # n_neighbors: Number of neighbors
        'KNM': KNeighborsClassifier(n_neighbors = 5),

        # Kernel: kernel type to be used in the algorithm, 
        # C: Regularization parameter
        'Linear SVM': SVC(kernel='linear', C=0.04),

        # Gamma: Kernel coefficient for RBF, C: Regularization parameter
        'SVM RBF': SVC(gamma = 0.01, C = 100),

        # Kernel: The covariance function of the GP
        'GP': GaussianProcessClassifier(kernel = (1 * RBF(1.0))),

        # Max_depth: The maximum depth of the tree
        # Max_features: The number of features to consider when looking for 
        #   the best split
        # Min_samples_split: The minimum number of samples required to split an 
        #   internal node:
        # N_estimators: The number of trees in the forest
        'Random Forest': RandomForestClassifier(
            max_depth = 3, max_features = 3, min_samples_split = 8, 
            n_estimators = 39),

        # Min_samples_split: The minimum number of samples required to split an 
        #   internal node
        # Min_sampes_leaf: The minimum number of samples required to be at a 
        #   leaf node
        # Max_depth: The maximum depth of the tree
        'DT': DecisionTreeClassifier(min_samples_split = 10, 
            min_samples_leaf = 10, max_depth = 6),

        # Alpha: regularization term
        # Max_iter: Maximum number of iterations
        'MLPC': MLPClassifier(alpha = 1.44, max_iter = 1000),

        # AdaBoost classifier: Classifier is a meta-estimator
        # N_estimators = The maximum number of estimators at which boosting is 
        #   terminated
        # Learning_rate = Weight applied to each classifier at each boosting 
        #   iteration
        'AdaBoost': AdaBoostClassifier(n_estimators = 4, learning_rate = 1),

        # Gaussian Naive Bayes
        'Naive Bayes': GaussianNB(),

        # Logistic Regression Classifier
        'Logistic Reg': LogisticRegression(),
        }

    # Try to read the dataset
    try:
        data = pd.read_csv(
            'Heart Failure Prediction Dataset.csv')

    except Exception as wrongValue:
        print('The file CSV cannot be opened')

    # Clean for wrog or invalid values from the dataset
    # Drop any row from the dataset that the value is Null
    data = data.dropna(axis=0, how='any')

    # Split Inputs and Outputs
    # For DataSet Heart Failure Prediction Dataset
    columns_output = ['HeartDisease']

    data['Sex'] = data['Sex'].apply(parse_sex)
    data['ChestPainType'] = data['ChestPainType'].apply(
        parse_chest_pain)
    data['RestingECG'] = data['RestingECG'].apply(
        parse_restingECG)
    data['ExerciseAngina'] = data['ExerciseAngina'].apply(
        parse_exerciseAngina)
    data['ST_Slope'] = data['ST_Slope'].apply(
        parse_stSlope)

    x = np.asanyarray(data.drop(columns=columns_output))[:,:]
    y = np.asanyarray(data[columns_output])[:].ravel()

    # Split into training, and test 
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.8)

    # Normalize values from the dataset
    x_train = StandardScaler().fit_transform(x_train)
    x_test = StandardScaler().fit_transform(x_test)

    # Foreach classifier
    for classifier in classifiers:
        # Verify if the model was already train
        if os.path.isfile('./' + classifier + '.pkl'):
            file = open('./' + classifier + '.pkl', 'rb')
            model = pickle.load(file)

            pred = model.predict(x_test)

        else:
            # Instantiate model to train
            model = classifiers[classifier]
            model.fit(x_train, y_train)

            pred = model.predict(x_test)

            # Save the trained model
            pickle.dump(model, open( classifier +'.pkl', 'wb'))

        print('Model: ', classifier)
        print(str(classifiers[classifier]))
        print('Train: ', model.score(x_train, y_train))
        print('Test: ', model.score(x_test, y_test))
        print('Accuracy: ', accuracy_score(y_test, pred))
        print('Clasification report: \n', classification_report(
            y_test, pred))
        print('Confusion matrix: \n', confusion_matrix(
            y_test, pred))

if __name__ == '__main__':
    main()
