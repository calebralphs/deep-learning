"""
How to Use:
    COVID DATASET
    1. Download Dataset: https://github.com/ieee8023/covid-chestxray-dataset
    2. Make sure you have a folder called covid-chestxray-dataset
    3. Move all images into folder called covid-chestxray-dataset/images
    4. Put the metadata in covid-chestxray-dataset

    NORMAL AND PNEUMONIA DATASET
    1. Download Dataset:https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
    2. Create two folders: normal, pneumonia
    3. Copy all normal / pneumonia: test, validation, and train images into respective folders

"""

from PIL import Image
import torchxrayvision as xrv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import glob
import copy

# this takes forever, shink size so faster while developing
#IMAGE_HEIGHT = 1400
#IMAGE_WIDTH = 1260

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 256

NUM_IMAGES_NORMAL = 300
NUM_IMAGES_PNEUMONIA = 300

PERCENT_TRAINING = 80
PERCENT_VALIDATION = 20  # percent of training set

# takes an array of images scaled between 0 and 1
def showMyImage(im):
    im = Image.fromarray(im * 255)
    im.show()

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

'''
Generate numpy array from covid images
Add labels for with key: [normal, covid19, pneumonia]
'''
def gen_covid_np():
    # Create data set:
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath="./covid-chestxray-dataset/images",
                                             csvpath="./covid-chestxray-dataset/metadata.csv")
    X = []
    Y = []
    for i in tqdm(range(len(d_covid19))):
        if d_covid19[i]['lab'][2] == 1:
            # retrieve image
            im = d_covid19[i]['PA'][0]

            # change scale from -1024:1024 to 0:255
            im = ((im+1024) / 2048) * 255

            # set to a constant image size
            im = Image.fromarray(im)
            im = im.resize((IMAGE_HEIGHT, IMAGE_WIDTH))

            # add image and label to array
            X.append(np.array(im))
            if d_covid19[i]['lab'][-1] or d_covid19[i]['lab'][-4]:
                Y.append([0, 1, 1])
            else:
                Y.append([0, 1, 0])

    np.save('covid_labels', np.array(Y))
    np.save('covid_scans', np.array(X))


'''
Generate numpy array from normal images
Add labels for with key: [normal, covid19, pneumonia]
'''
def gen_normal_np():
    os.chdir("./normal")
    images = glob.glob("*.jpeg")
    X = []
    Y = []
    for image in images:
        if len(X) < NUM_IMAGES_NORMAL:
            im = Image.open(image).convert('L')
            im = im.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
            X.append(np.array(im))
            Y.append([1, 0, 0])
    os.chdir("../")
    np.save('normal_scans', np.array(X))
    np.save('normal_labels', np.array(Y))


'''
Generate numpy array from pneumonia images
Add labels for with key: [normal, covid19, pneumonia]
'''
def gen_pneumonia_np():
    os.chdir("./pneumonia")
    images = glob.glob("*.jpeg")
    X = []
    Y = []
    for image in images:
        if len(X) < NUM_IMAGES_PNEUMONIA:
            if "virus" in image:
                im = Image.open(image).convert('L')
                im = im.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
                X.append(np.array(im))
                Y.append([0, 0, 1])
    os.chdir("../")
    np.save('pneumonia_scans', np.array(X))
    np.save('pneumonia_labels', np.array(Y))

def create_training_images(X, y):
    X_reflect = np.flip(copy.deepcopy(X), axis=1)
    y_reflect = copy.deepcopy(y)

    image_noise = copy.deepcopy(X)
    for i in range(len(image_noise.T[0])):
        random_noise = .1 * np.random.rand(np.shape(image_noise)[1], np.shape(image_noise)[2], np.shape(image_noise)[3])
        image_noise[i] = image_noise[i] + random_noise
    y_noise = copy.deepcopy(y)

    X_reflect_noise = np.flip(copy.deepcopy(image_noise), axis=1)
    y_reflect_noise = copy.deepcopy(y)

    X = np.vstack((X, X_reflect))
    y = np.vstack((y, y_reflect))

    X = np.vstack((X, image_noise))
    y = np.vstack((y, y_noise))

    X = np.vstack((X, X_reflect_noise))
    y = np.vstack((y, y_reflect_noise))

    seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    return X, y

# generate the np arrays
#gen_covid_np()
#gen_normal_np()
#gen_pneumonia_np()

# load arrays and generate image and label arrays
X_normal = np.expand_dims(np.load('normal_scans.npy', allow_pickle=True), axis=3)
Y_normal = np.load('normal_labels.npy', allow_pickle=True)
X_covid = np.expand_dims(np.load('covid_scans.npy', allow_pickle=True), axis=3)
Y_covid = np.load('covid_labels.npy', allow_pickle=True)
X_pneumonia = np.expand_dims(np.load('pneumonia_scans.npy', allow_pickle=True), axis=3)
Y_pneumonia = np.load('pneumonia_labels.npy', allow_pickle=True)

X = np.vstack((X_normal, X_pneumonia))
y = np.vstack((Y_normal, Y_pneumonia))
X = np.vstack((X, X_covid))
y = np.vstack((y, Y_covid))

print("Normal Images: ", np.shape(X_normal)[0])
print("Covid Images: ", np.shape(X_covid)[0])
print("Pneumonia Images: ", np.shape(X_pneumonia)[0], '\n')


seed = np.random.randint(0, 10000)
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(y)

total_images = np.shape(X)[0]
num_training = int(total_images * PERCENT_TRAINING / 100)
num_validation = int(num_training * PERCENT_VALIDATION / 100)

X_tr = X[0:num_training] / 255
X_val, X_tr = X_tr[0:num_validation], X_tr[num_validation:]
X_te = X[num_training:] / 255

y_tr = y[0:num_training]
y_val, y_tr = y_tr[0:num_validation], y_tr[num_validation:]
y_te = y[num_training:]

'''
Add some data augmentation here!
1. Flipping
2. Gaussian Noise
'''
X_tr, y_tr = create_training_images(X_tr, y_tr)


print("Training Images: ", np.shape(X_tr)[0])
print("Validation Images: ", np.shape(X_val)[0])
print("Testing Images: ", np.shape(X_te)[0], '\n')


# DEFINE MODEL
def create_model():
    model = keras.Sequential()  # Must define the input shape in the first layer of the neural network
    model.add(
        keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(3, activation='sigmoid'))
    return model

model = create_model()
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#model.summary()

LOAD_WEIGHTS = True

if LOAD_WEIGHTS == False:
    # TRAIN
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=32, epochs=10)
    model.save_weights("covid-19-model.h5")

    # PLOT RESULTS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    plt.grid(True)
    plt.show()

else:
    model.load_weights("covid-19-model.h5")


# TEST
score = model.evaluate(X_te, y_te, verbose=0)  # Print test accuracy
print('\n', 'Test accuracy:', score[1])


label_guide = ['normal', 'covid', 'pneumonia']

'''
# plot some examples and their predictions
examples = 5
for i in range(examples):
    im = X_te[i,:,:,0]
    label = y_te[i]
    plt.imshow(im)
    prediction = np.round(model.predict(np.expand_dims(X_te[i], axis=0), batch_size=1))[0]
    plt.title('Ground Truth: ' + str(label) +
              '\n' + str(np.take(label_guide, np.nonzero(label)[0]))+
              '\nPrediction: ' + str(prediction) +
              '\n' + str(np.take(label_guide, np.nonzero(prediction)[0])))
    plt.tight_layout()
    plt.show()
'''

results_dict = {
                "normal": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0, "covid": 0, "other": 0},
                "pneumonia": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0, "covid": 0, "other": 0},
                "covid-pneumonia": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0, "covid": 0, "other": 0},
                "covid": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0, "covid": 0, "other": 0}
                }
for i, img in enumerate(X_te):
    label = y_te[i]
    prediction = np.round(model.predict(np.expand_dims(X_te[i], axis=0), batch_size=1))[0]
    result_true = np.take(label_guide, np.nonzero(label)[0])
    result_predict = np.take(label_guide, np.nonzero(prediction)[0])

    if len(result_true) > 1:
        if result_true[0] == 'covid' and result_true[1] == 'pneumonia':
            result_true = ['covid-pneumonia']
        else:
            result_true = ["other"]
    if len(result_predict) > 1:
        if result_predict[0] == 'covid' and result_predict[1] == 'pneumonia':
            result_predict = ['covid-pneumonia']
        else:
            result_predict = ["other"]
    if len(result_predict) == 0:
        result_predict = ["other"]

    a = results_dict[result_true[0]]
    a[result_predict[0]] += 1

# 4 groups, 5 bars per group
data = np.zeros((4, 3))
for key, val in results_dict.items():
    if key == 'normal':
        c = 0
    elif key == 'pneumonia':
        c = 1
    else:
        c = 2

    for key2, val in val.items():
        if key2 == 'normal':
            r = 0
        elif key2 == 'pneumonia':
            r = 1
        elif key2 == 'covid-pneumonia':
            r = 2
        else:
            r = 3
        pred = results_dict[key]
        data[r][c] += pred[key2]

print(data)

X = np.arange(3)*1.2
plt.bar(X + 0.0, data[0], color='lightsteelblue', width=0.25, label='normal')
plt.bar(X + 0.25, data[1], color='forestgreen', width=0.25, label='pneumonia')
plt.bar(X + 0.5, data[2], color='cyan', width=0.25, label='covid-pneumonia')
plt.bar(X + 0.75, data[3], color='saddlebrown', width=0.25, label='other')

plt.ylabel('Number of Occurances')
plt.title('Predictions: label vs prediction')
plt.xticks(X + 0.6, ('Normal', 'Pneumonia', 'Covid-Pneumonia'))
plt.yticks(np.arange(0, 80, 5))

plt.legend()

plt.show()
