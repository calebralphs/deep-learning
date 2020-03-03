import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import keras


def showImage(image):
    # Show an arbitrary test image in grayscale
    fig,ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    plt.show()
'''
faces = np.load("./facesAndAges/faces.npy")
ages = np.load("./facesAndAges/ages.npy")

x_train, x_valid, x_test = faces[2500:], faces[1250:2500], faces[0:1250]
y_train, y_valid, y_test = ages[2500:], ages[1250:2500], ages[:1250]

# training images
for i in range(len(x_train)):
    im = Image.fromarray(x_train[i])
    im = im.resize((240, 240))
    im = im.convert('RGB')
    path = "./facesAndAges/images/train/"
    if not os.path.exists(path+str(y_train[i])):
        os.mkdir(path+str(y_train[i]))
    name = path + str(y_train[i]) + "/face"+str(i)+".jpg"
    im.save(name)

# validation images
for i in range(len(x_valid)):
    im = Image.fromarray(x_valid[i])
    im = im.resize((240, 240))
    im = im.convert('RGB')
    path = "./facesAndAges/images/valid/"
    if not os.path.exists(path+str(y_valid[i])):
        os.mkdir(path+str(y_valid[i]))
    name = path + str(y_valid[i]) + "/face"+str(i)+".jpg"
    im.save(name)

# testing images
for i in range(len(x_test)):
    im = Image.fromarray(x_test[i])
    im = im.resize((240, 240))
    im = im.convert('RGB')
    path = "./facesAndAges/images/test/"
    if not os.path.exists(path+str(y_test[i])):
        os.mkdir(path+str(y_test[i]))
    name = path + str(y_test[i]) + "/face"+str(i)+".jpg"
    im.save(name)

'''

train_dir = "./facesAndAges/images/train"
validation_dir = "./facesAndAges/images/valid"
test_dir = "./facesAndAges/images/test"
image_size = 48

train_datagen = keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255,
      rotation_range=5,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 64
val_batchsize = 32

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=train_batchsize,
        class_mode='sparse')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='sparse',
        shuffle=False)

test_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=val_batchsize,
        class_mode='sparse',
        shuffle=False)

model = keras.Sequential()
max_value = 100

# Add new layers
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(image_size,image_size,1)))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu'))
model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=None, padding='valid', data_format=None))
model.add(keras.layers.Dense(256))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(128))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(64))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(32))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(16))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(8))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(4))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(2))
model.add(keras.layers.ReLU(max_value=max_value))
model.add(keras.layers.Dense(1))
model.summary()

EPOCHS = 10
LEARNING_RATE = .001

sgd = keras.optimizers.SGD(lr=.0001)

# Compile the model
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['mae', 'mse'])

history = model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=train_generator.samples/train_generator.batch_size, #20
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples/validation_generator.batch_size,
                    verbose=1)


loss = model.evaluate_generator(test_generator, steps=100)
print(loss)