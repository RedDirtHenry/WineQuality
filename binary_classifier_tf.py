import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams

if __name__ == "__main__":
    win_data = pd.read_csv("/home/jtotiker/Documents/DataScience/Projects/LinReg/data/winsorized_data.csv")
    print(win_data.shape)


    # To get to a binary classification wines that are quality 6 or higher will be considered of high quality and wines
    #  that are 5 or lower will be considered low quality. This way the goal is to predict simply if a wine is high or low quality
    win_data['high_quality'] = [
    1 if quality >= 6 else 0 for quality in win_data['quality']
    ]

    # Drop quality and split data into train and test using an 80/20 split
    win_data.drop('quality', axis=1, inplace=True)

    X = win_data.drop('high_quality', axis=1)
    y = win_data['high_quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, random_state=42
    )

    # Same as for linear regression model will over sample to account for data set imbalance
    oversample = RandomOverSampler(random_state=88)

    X_train, y_train = oversample.fit_resample(X_train, y_train)
        
    # Rescale Data just as was done for linear regression model using StandardScaler
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Construct Neural Network
    '''
        Loss Function: binary cross-entropy
        Evaluation Metric: accuracy, precision, and recall

        As it has been seen that our dataset is unbalanced, precision and recall will be important to view with accuracy to
        evaluate the model's performance as well.

        Input Layer: Dictated by number of features inputted
        Hidden Layers: One hidden layer should be sufficient given the low complexity of
            the problem. Following rule of thumb, will go for the ratio of the number of samples
            over input + output neurons times a scaling factor of 2, which puts us close to a typical
            number of 256 neurons. Given the small sample size, this will also help to avoid
            overfitting with two many hidden layers and neurons.
        Output Layer: One Neuron, our output for classifying high/low quality wine (> 0.5 high, or ≤ 0.5 low)

        Note: Some testing was done with up to 3 hidden layers of various sizes, and there was no marked
        improvement in any of the evaluation metrics, typically they had spikes of bad convergence, supporting
        that one layer is the best approach for this problem.
    '''
    tf.random.set_seed(88)
    model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.03),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    e = 300
    history = model.fit(X_train, y_train, epochs=e, verbose=0)


    # Plotting Model Evaluation Metrics
    rcParams['figure.figsize'] = (12, 8)


    plt.plot(
    np.arange(1, e+1), 
    history.history['loss'], label='Loss'
    )
    plt.plot(
        np.arange(1, e+1), 
        history.history['accuracy'], label='Accuracy'
    )
    plt.plot(
        np.arange(1, e+1), 
        history.history['precision'], label='Precision'
    )
    plt.plot(
        np.arange(1, e+1), 
        history.history['recall'], label='Recall'
    )
    plt.title('Model Evaluation Metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    plt.show()

    # Using Model to Predict Wine Qualities of Test Data Set
    predict_probability = model.predict(X_test)

    quality = predict_probability.flatten()
    n = -1
    for i in quality:
        n = n + 1
        if i > 0.5:
            quality[n] = 1
        else:
            quality[n] = 0

    print(f'Accuracy: {100*accuracy_score(y_test, quality):.2f}%')
    print(f'Precision: {100*precision_score(y_test, quality):.2f}%')
    print(f'Recall: {100*recall_score(y_test, quality):.2f}%')

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, quality)

    ax= plt.subplot()
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="RdBu", ax=ax )

    # labels, title and ticks
    ax.set_xlabel('Predicted Wine Quality');ax.set_ylabel('True Wine Quality'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['High', 'Low']); ax.yaxis.set_ticklabels(['High', 'Low'])