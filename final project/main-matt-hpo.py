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

from matt_generate_data import *
from matt_keras_model import *
from matt_graph_generation import *

####################################################################
# DEFINE CONSTANT VARIABLES

IMAGE_HEIGHT = 288  # 576
IMAGE_WIDTH = 256  # 512

NUM_IMAGES_NORMAL = 300
NUM_IMAGES_PNEUMONIA = 300

PERCENT_TRAINING = 80
PERCENT_VALIDATION = 20  # percent of training set

SHUFFLE_SEED = 100  # if you change the seed, you must retrain the network and delete all saved_models
LABEL_GUIDE = ['normal', 'covid-pneumonia', 'pneumonia']
####################################################################



####################################################################
# DEFINE TRAINING VARIABLES

LOAD_WEIGHTS = True
SAVE_BEST = True

# 'normal', 'distill', 'smooth'
LABEL_FORMAT = 'normal'
TEMPERATURE = 1.0
SMOOTH_FACTOR = .1

NUM_EPOCHS = 20
BATCH_SIZE = 32

####################################################################



####################################################################
# GENERATE DATA

# generate the np arrays
generate_np_arrays(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_NORMAL, NUM_IMAGES_PNEUMONIA, gen_normal=False, gen_pneumonia=False, gen_covid=False)
X_tr, y_tr, X_val, y_val, X_te, y_te = generate_train_val_test_sets(LABEL_FORMAT, PERCENT_TRAINING, PERCENT_VALIDATION, SHUFFLE_SEED, SMOOTH_FACTOR);

####################################################################





####################################################################
# CREATE AND TEST MODEL

model = create_model(LABEL_FORMAT, IMAGE_WIDTH, IMAGE_HEIGHT, TEMPERATURE)
model.compile(loss='categorical_crossentropy', #binary_crossentropy
              optimizer='adam',
              metrics=['accuracy'])
#model.summary()

if LOAD_WEIGHTS:
    # default load weight
    load_weights(model, LABEL_FORMAT)

    # load a custom weight
    model.load_weights("saved_models/covid-19-model-distilled.h5")
else:
    current_dt = datetime.datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
    history = model.fit(X_tr, y_tr,
                        validation_data=(X_val, y_val),
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        callbacks=[CustomCallback(model, X_tr, y_tr, X_val, y_val, X_te, y_te, SAVE_BEST, current_dt)]
                        )
    if SAVE_BEST:
        plt.savefig("accuracyVsEpochs/Training_Accuracy_vs_Epochs_" + LABEL_FORMAT + "_batchsize" + str(BATCH_SIZE) + "_" + current_dt +".png")
    plt.clf()
    plt.ioff()
    plot_loss_during_training(history)
    save_weights(model, LABEL_FORMAT, X_tr, X_val)


# Test Model
score = model.evaluate(X_te, y_te, verbose=0)  # Print test accuracy
print('\n', 'Test accuracy:', score[1])
####################################################################




####################################################################
# plot the results of the testing / training

#plot_images_and_predictions(model, label_guide, X_te, y_te);
plot_distribution_of_predictions(model, LABEL_GUIDE, X_te, y_te)

####################################################################

