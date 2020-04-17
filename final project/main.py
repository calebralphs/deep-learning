"""
a sample d_covid19[0] has photo data d_covid19[0]['PA'], 14-length one hot vector label d_covid19['lab'],
and integer index d_covid19['idx']
A = posteroanterior,AP = anteroposterior, AP Supine = laying down


label guide:
{0 'ARDS': {0.0: 187, 1.0: 14},
 1 'Bacterial Pneumonia': {0.0: 184, 1.0: 17},
 2 'COVID-19': {0.0: 46, 1.0: 155},
 3 'Chlamydophila': {0.0: 200, 1.0: 1},
 4 'Fungal Pneumonia': {0.0: 188, 1.0: 13},
 5 'Klebsiella': {0.0: 200, 1.0: 1},
 6 'Legionella': {0.0: 199, 1.0: 2},
 7 'MERS': {0.0: 201},
 8 'No Finding': {0.0: 200, 1.0: 1},
 9 'Pneumocystis': {0.0: 188, 1.0: 13},
 10 'Pneumonia': {0.0: 1, 1.0: 200},
 11 'SARS': {0.0: 190, 1.0: 11},
 12 'Streptococcus': {0.0: 188, 1.0: 13},
 13 'Viral Pneumonia': {0.0: 35, 1.0: 166}}
"""




import torchxrayvision as xrv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import keras

def dict_to_np():
    # Create data set:
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath="./covid-chestxray-dataset/images",
                                             csvpath="./covid-chestxray-dataset/metadata.csv")
    X = []
    Y = []
    for i in tqdm(range(len(d_covid19))):
        X.append(d_covid19[i]['PA'][0])
        Y.append(d_covid19[i]['lab'])
    np.save('labels', np.array(Y))
    np.save('CT_scans', np.array(X))


'''
smallest image is 156 rows x 157 columns pixels
'''
def dict_to_center_cropped_np():
    new_width = 156
    new_height = 156

    # Create data set:
    d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                             imgpath="./covid-chestxray-dataset/images",
                                             csvpath="./covid-chestxray-dataset/metadata.csv")
    X = []
    Y = []
    for i in tqdm(range(len(d_covid19))):
        im = d_covid19[i]['PA'][0]
        width = len(im[0])
        height = len(im)

        left = int((width - new_width) / 2)
        top = int((height + new_height) / 2)
        right = int((width + new_width) / 2)
        bottom = int((height - new_height) / 2)

        # Crop the center of the image
        im = im[bottom:top, left:right]
        X.append(im)
        Y.append(d_covid19[i]['lab'])

    np.save('center_cropped_labels', np.array(Y))
    np.save('center_cropped_CT_scans', np.array(X))




label_guide = ['ARDS','Bacterial Pneumonia','COVID-19','Chlamydophila', 'Fungal Pneumonia', 'Klebsiella', 'Legionella',
               'MERS', 'No Finding', 'Pneumocystis', 'Pneumonia', 'SARS', 'Streptococcus', 'Viral Pneumonia']
# Load data set:
X = np.load('center_cropped_CT_scans.npy', allow_pickle=True)
# 14-length multi-class one hot. Example: [0,0,1,0,1,0,0,0,1,0,0,0,0,0]. Multiple outcomes can be true for one sample
Y = np.load('center_cropped_labels.npy', allow_pickle=True)

# DATA PRE-PROCESSING
# data is somewhat grouped by label so we must shuffle them. X and Y must be shuffled with same seed!
np.random.seed(42)
np.random.shuffle(X)
np.random.seed(42)
np.random.shuffle(Y)

# len(X) = 242. 8:1:1 split
X_tr = np.expand_dims(X[0:194].astype('float32') / 255, axis=-1)  # 194
X_val = np.expand_dims(X[194:218].astype('float32') / 255, axis=-1)  # 24
X_t = np.expand_dims(X[218:].astype('float32') / 255, axis=-1)  # 24
y_tr = Y[0:194]  # 194
y_val = Y[194:218]  # 24
y_t = Y[218:]  # 24


# DEFINE MODEL
height = len(X_tr[0])
width = len(X_tr[0,0])

model = keras.Sequential()  # Must define the input shape in the first layer of the neural network
model.add(
    keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(height, width, 1)))
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
model.add(keras.layers.Dense(14, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.summary()


# TRAIN
history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), batch_size=1, epochs=10)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
plt.grid(True)
plt.show()


# TEST
score = model.evaluate(X_t, y_t, verbose=0)  # Print test accuracy
print('\n', 'Test accuracy:', score[1])

# plot some examples and their predictions
examples = 5
for i in range(examples):
    im = X_t[i,:,:,0]
    label = y_t[i]
    plt.imshow(im)
    prediction = np.round(model.predict(np.expand_dims(X_t[i], axis=0), batch_size=1))[0]
    plt.title('Ground Truth: ' + str(label) +
              '\n' + str(np.take(label_guide, np.nonzero(label)[0] ))+
              '\nPrediction: ' + str(prediction) +
              '\n' + str(np.take(label_guide, np.nonzero(prediction)[0] )) )
    plt.tight_layout()
    plt.show()


