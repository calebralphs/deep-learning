from PIL import Image
import torchxrayvision as xrv
from tqdm import tqdm
import numpy as np
import os
import glob
import copy

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
def gen_covid_np(IMAGE_HEIGHT, IMAGE_WIDTH):
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
                Y.append([0, 1, 0])

    np.save('np_arrays/covid_labels', np.array(Y))
    np.save('np_arrays/covid_scans', np.array(X))


'''
Generate numpy array from normal images
Add labels for with key: [normal, covid19, pneumonia]
'''
def gen_normal_np(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_NORMAL):
    os.chdir("datasets/normal")
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
    np.save('np_arrays/normal_scans', np.array(X))
    np.save('np_arrays/normal_labels', np.array(Y))


'''
Generate numpy array from pneumonia images
Add labels for with key: [normal, covid19, pneumonia]
'''
def gen_pneumonia_np(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_PNEUMONIA):
    os.chdir("datasets/pneumonia")
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
    np.save('np_arrays/pneumonia_scans', np.array(X))
    np.save('np_arrays/pneumonia_labels', np.array(Y))


def generate_np_arrays(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_NORMAL, NUM_IMAGES_PNEUMONIA, gen_normal=False, gen_pneumonia=False, gen_covid=False):
    if gen_normal:
        gen_normal_np(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_NORMAL)
    if gen_pneumonia:
        gen_pneumonia_np(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_IMAGES_PNEUMONIA)
    if gen_covid:
        gen_covid_np(IMAGE_HEIGHT, IMAGE_WIDTH)


def create_training_images(X, y):
    X_reflect = np.flip(copy.deepcopy(X), axis=1)
    y_reflect = copy.deepcopy(y)

    image_noise = copy.deepcopy(X)
    for i in range(np.shape(image_noise)[0]):
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

def generate_train_val_test_sets(LABEL_FORMAT, PERCENT_TRAINING, PERCENT_VALIDATION, SHUFFLE_SEED, SMOOTH_FACTOR):
    # load arrays and generate image and label arrays
    X_normal = np.expand_dims(np.load('np_arrays/normal_scans.npy', allow_pickle=True), axis=3)
    Y_normal = np.load('np_arrays/normal_labels.npy', allow_pickle=True)
    X_covid = np.expand_dims(np.load('np_arrays/covid_scans.npy', allow_pickle=True), axis=3)
    Y_covid = np.load('np_arrays/covid_labels.npy', allow_pickle=True)
    X_pneumonia = np.expand_dims(np.load('np_arrays/pneumonia_scans.npy', allow_pickle=True), axis=3)
    Y_pneumonia = np.load('np_arrays/pneumonia_labels.npy', allow_pickle=True)

    X = np.vstack((X_normal, X_pneumonia))
    y = np.vstack((Y_normal, Y_pneumonia))

    X = np.vstack((X, X_covid))
    y = np.vstack((y, Y_covid))

    print("Total Normal Images: ", np.shape(X_normal)[0])
    print("Total Covid Images: ", np.shape(X_covid)[0])
    print("Total Pneumonia Images: ", np.shape(X_pneumonia)[0], '\n')

    seed = SHUFFLE_SEED
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

    print("Total Training Images: ", np.shape(X_tr)[0])
    print("Total Validation Images: ", np.shape(X_val)[0])
    print("Total Testing Images: ", np.shape(X_te)[0], '\n')


    # the distilled arrays are updated every time we train the model in "normal" mode
    if LABEL_FORMAT == 'distill':
        y_tr = np.load("np_arrays/y_tr_distilled.npy")
        y_val = np.load("np_arrays/y_val_distilled.npy")
    if LABEL_FORMAT == 'smooth':
        y_tr = smooth_labels(y_tr, SMOOTH_FACTOR)
        y_val = smooth_labels(y_val, SMOOTH_FACTOR)

    return X_tr, y_tr, X_val, y_val, X_te, y_te


'''
Smooths labels as benchmark for distillation
'''
def smooth_labels(labels, factor=0.15):
    # smooth the labels
    labels = labels*(1 - factor)
    labels += (factor / labels.shape[1])
    # returned the smoothed labels
    return labels


def save_distilled_y(model, X, name):
    prediction = []
    for i in range(np.shape(X)[0]):
        prediction.append(model.predict(np.expand_dims(X[i], axis=0), batch_size=1)[0])

    np.save(name, prediction)
