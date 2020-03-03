import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
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
image_size = 224

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
train_batchsize = 20
val_batchsize = 10

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


vgg_model = VGGFace(model='senet50', include_top=False, input_shape=(image_size, image_size, 3), pooling='avg')

for i in range(20):
    vgg_model.layers.pop()


for layer in vgg_model.layers[:-3]:
    layer.trainable = False

model = keras.Sequential()
model.add(vgg_model)

# Add new layers
model.add(keras.layers.Dense(500))
model.add(keras.layers.ReLU(max_value=100))
model.add(keras.layers.Dense(1))
model.summary()

EPOCHS = 100
LEARNING_RATE = .001

sd=[]
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [1,1]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        sd.append(step_decay(len(self.losses)))
        print('lr:', step_decay(len(self.losses)))

def step_decay(losses):
    if h.losses[-1] < 200:
        lrate = 0.0
        return lrate
    elif h.losses[-1] < 500:
        lrate=0.0000000001
        return lrate
    elif h.losses[-1] < 3000:
        lrate=0.00001
        return lrate
    else:
        lrate=0.0001
        return lrate

h = LossHistory()
lrate = keras.callbacks.LearningRateScheduler(step_decay)


sgd = keras.optimizers.SGD(lr=.0001)

# Compile the model
model.compile(loss='mse',
              optimizer=sgd,
              metrics=['mae', 'mse'])

history = model.fit_generator(train_generator,
                    epochs=EPOCHS,
                    steps_per_epoch=20, #train_generator.samples/train_generator.batch_size,
                    callbacks=[h, lrate],
                    verbose=1)


loss = model.evaluate_generator(test_generator, steps=100)
print(loss)

print(history)


#model.load_weights('model.weights.best.hdf5')



# plot model fit over each epoch
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

