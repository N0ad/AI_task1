import numpy as np
import cv2
import keyboard
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import to_categorical
from tensorflow.keras import backend as K


def custom_f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        recall = TP / (Positives+K.epsilon())
        return recall


    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

        precision = TP / (Pred_Positives+K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+K.epsilon()))

np.random.seed(20)

mode = "test"

with open("./annotation.txt") as file:
    lines = [line.rstrip() for line in file]

np.random.shuffle(lines)

def parser(line):
    args = line.split(',')
    return args

x_data = []
y_data = []
labels = []
labels_count = 0

for line in lines:
    args = parser(line)
    image = cv2.imread(args[0])
    if (labels == []):
        labels_count = labels_count + 1
        labels.append(args[5])
        y_data.append(np.array(0))
    else:
        i = 0
        num = 0
        for label in labels:
            if (label == args[5]):
                i = 1
                break
            num = num + 1
        if (i == 0):
            labels.append(args[5])
            labels_count = labels_count + 1
        y_data.append(np.array(num))
    image = cv2.resize(image, (64, 64))
    x_data.append(image)

length = int(len(x_data) * 0.8)
x_train = np.array(x_data[:length])
y_train = np.array(y_data[:length])
x_valid = np.array(x_data[length:])
y_valid = np.array(y_data[length:])
y_train = to_categorical(y_train, num_classes=labels_count)
y_valid = to_categorical(y_valid, num_classes=labels_count)

if (mode == "teach"):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu', padding = 'same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Dropout(0))
    model.add(BatchNormalization())
    model.add(Flatten())
    
    model.add(BatchNormalization())
    model.add(Dense(labels_count))
    model.add(Activation('softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'mean_squared_error', 'mean_absolute_error', custom_f1])
    model = load_model('./model10.h5', custom_objects={"custom_f1": custom_f1})
    model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = 1, batch_size = 64, shuffle = True)
    model.save('model10.h5')

if (mode == "test"):
    model = load_model('./model10.h5', custom_objects={"custom_f1": custom_f1})

def convertPrediction(labels, length):
    max = 0
    num = -1
    for i in range(0, length):
        if (max < labels[i]):
            max = labels[i]
            num = i
    return num

statistic = [[0, 0, 0, 0]]
for i in range(1, labels_count):
    statistic.append([0, 0, 0, 0])

x_test = []
y_test = []
with open("./test_annotation.txt") as file:
    lines = [line.rstrip() for line in file]

images = []

for line in lines:
    args = parser(line)
    image = cv2.imread(args[0])
    images.append(image)
    i = 0
    num = 0
    for label in labels:
        if (label == args[5]):
            break
        num = num + 1
    y_test.append(np.array(num))
    x1 = int(args[1])
    y1 = int(args[2])
    x2 = int(args[3])
    y2 = int(args[4])
    if ((y2 - y1 > 15) & (x2 - x1 > 15)):
        image = image[y1:y2, x1:x2]
    image = cv2.resize(image, (64, 64))
    x_test.append(image)

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = to_categorical(y_test, num_classes=labels_count)
length = len(x_test)
predicts = []

for i in range(0, length):
    tst = np.array([x_test[i]])
    res = model.predict(tst, batch_size= 64)
    real_label = convertPrediction(y_test[i], labels_count)
    predict_label = convertPrediction(res[0], labels_count)
    predicts.append(predict_label)
    statistic[real_label][0] = statistic[real_label][0] + 1
    if (real_label == predict_label):
        statistic[real_label][1] = statistic[real_label][1] + 1
    else:
        statistic[real_label][2] = statistic[real_label][2] + 1
        statistic[predict_label][3] = statistic[predict_label][3] + 1

print(model.summary())

scores = model.evaluate(x_test, y_test, verbose = 0)

print("Loss: ")
print(scores[0])
print("Accuracy: ")
print(scores[1])
print("MSE: ")
print(scores[2])
print("MAE: ")
print(scores[3])
print("F1: ")
print(scores[4])

for i in range(0, labels_count):
    print(labels[i] + ": Total images - " + str(statistic[i][0]) + " | Correctly defined - " + str(statistic[i][1]) + " | Incorrectly defined " + str(statistic[i][2]) + " | Impostors: " + str(statistic[i][3]))

i = 0
while (1):
    tst = [x_test[i]]
    cv2.imshow("32x32", cv2.resize(tst[0], (1024, 1024)))
    real_label = labels[convertPrediction(y_test[i], labels_count)]
    predict_label = labels[predicts[i]]
    cv2.imshow("Real: " + real_label + " | Predict: " + predict_label, cv2.resize(images[i], (1024, 1024)))
    button = cv2.waitKey(0) & 0xff
    if (button == ord('q')):
        cv2.destroyAllWindows()
        break
    if (button == ord('a')):
        i = i - 1
        if (i < 0):
            i = length - 1
        cv2.destroyAllWindows()
    if (button == ord('d')):
        i = i + 1
        if (i == length):
            i = 0
        cv2.destroyAllWindows()
