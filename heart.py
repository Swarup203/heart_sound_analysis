import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
#matplotlib.use('Agg')

# Paths to the dataset folders
#DATASET_PATH = 'D:/programs/project/Training_Data_CAD_2016/'
DATASET_PATH ='/home/divyaswarup/Training_Data_CAD_2016'
CATEGORIES = ['training-b-abnormal-2016', 'training-b-normal-2016', 'training-e-abnormal-2016', 'training-e-normal-2016']

# Parameters
SAMPLE_RATE = 2000
DURATION = 3
MFCC_FEATURES = 40
INPUT_SHAPE = (MFCC_FEATURES, 128, 1)

def load_wav_files(data_path, categories):
    X, Y = [], []
    for category in categories:
        folder_path = os.path.join(data_path, category)
        label = 1 if 'abnormal' in category else 0
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(folder_path, file_name)
                audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
                mfcc = np.resize(mfcc, (MFCC_FEATURES, 128))
                X.append(mfcc)
                Y.append(label)
    return np.array(X), np.array(Y)


# Load data
X, Y = load_wav_files(DATASET_PATH, CATEGORIES)

X = X[..., np.newaxis]

# Encode labels
Y = to_categorical(Y, num_classes=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# Build the CNN model
def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = create_model(INPUT_SHAPE)
model.summary()


# Train the model
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=30, batch_size=32)



# Plotting accuracy and loss graphs
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("accuracy.png")
    plt.show()

plot_history(history)



# Evaluate the model
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_test, axis=1)

    print("Classification Report:\n", classification_report(Y_true, Y_pred_classes))

    cm = confusion_matrix(Y_true, Y_pred_classes)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig("evaluation.png")
    plt.show()


evaluate_model(model, X_test, Y_test)



# Function to predict on a new WAV file
def predict_wav_file(file_path, model):
    audio_data, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=MFCC_FEATURES)
    mfcc = np.resize(mfcc, (MFCC_FEATURES, 128))
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mfcc)
    predicted_label = np.argmax(prediction)

    if predicted_label == 1:
        print(f"The predicted class for {file_path} is: Abnormal")
    else:
        print(f"The predicted class for {file_path} is: Normal")


# Predict unknown input
test_file = r'/home/divyaswarup/Training_Data_CAD_2016/training-e-abnormal-2016/e00020.wav'
#test_file = r'/home/divyaswarup/Training_Data_CAD_2016/training-e-abnormal-2016\e00020.wav''
predict_wav_file(test_file, model)



# Predict unknown input
test_file = r'/home/divyaswarup/Training_Data_CAD_2016/training-b-normal-2016/b0001.wav'
#test_file = r'/home/divyaswarup/Training_Data_CAD_2016/training-b-normal-2016\b0001.wav'
predict_wav_file(test_file, model)


#predict unknown INPUT
test_file = r'/home/divyaswarup/Training_Data_CAD_2016/training-e-normal-2016/e00055.wav'
#test_file =  r'/home/divyaswarup/Training_Data_CAD_2016/training-e-normal-2016/e00055.wav'
predict_wav_file(test_file, model)

