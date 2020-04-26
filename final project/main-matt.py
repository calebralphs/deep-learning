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

# this takes forever, shink size so faster while developing
#IMAGE_HEIGHT = 1400
#IMAGE_WIDTH = 1260

IMAGE_HEIGHT = 285
IMAGE_WIDTH = 256

NUM_IMAGES_NORMAL = 500
NUM_IMAGES_PNEUMONIA = 500

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

    # [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 1.]

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


'''
Add some data augmentation here!
1. Flipping
2. Gaussian Noise
'''


label_guide = ['Normal', 'Covid-19', 'Pneumonia']

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
    # model.add(tf.keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(3, activation='sigmoid'))
    return model

model = create_model()
model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()


# TRAIN
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=20, epochs=10)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.grid(True)
plt.show()


# TEST
score = model.evaluate(X_te, y_te, verbose=0)  # Print test accuracy
print('\n', 'Test accuracy:', score[1])

# plot some examples and their predictions
examples = 5
for i in range(examples):
    im = X_te[i,:,:,0]
    label = y_te[i]
    plt.imshow(im)
    prediction = np.round(model.predict(np.expand_dims(X_te[i], axis=0), batch_size=1))[0]
    plt.title('Ground Truth: ' + str(label) +
              '\n' + str(np.take(label_guide, np.nonzero(label)[0] ))+
              '\nPrediction: ' + str(prediction) +
              '\n' + str(np.take(label_guide, np.nonzero(prediction)[0] )) )
    plt.tight_layout()
    plt.show()


