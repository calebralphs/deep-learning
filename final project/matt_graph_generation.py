from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_during_training(history):
    # PLOT RESULTS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    plt.grid(True)
    plt.show()


def plot_images_and_predictions(model, label_guide, X_te, y_te):
    # plot some examples and their predictions
    examples = 5
    for i in range(examples):
        im = X_te[i, :, :, 0]
        label = y_te[i]
        plt.imshow(im)
        prediction = np.round(model.predict(np.expand_dims(X_te[i], axis=0), batch_size=1))[0]
        plt.title('Ground Truth: ' + str(label) +
                  '\n' + str(np.take(label_guide, np.nonzero(label)[0])) +
                  '\nPrediction: ' + str(prediction) +
                  '\n' + str(np.take(label_guide, np.nonzero(prediction)[0])))
        plt.tight_layout()
        plt.show()


def plot_distribution_of_predictions(model, label_guide, X_te, y_te):
    results_dict = {
                    "normal": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0},
                    "pneumonia": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0},
                    "covid-pneumonia": {"normal": 0, "pneumonia": 0, "covid-pneumonia": 0},
                    }
    for i, img in enumerate(X_te):
        label = y_te[i]
        labelIndex = np.where(label == np.amax(label))

        prediction = model.predict(np.expand_dims(X_te[i], axis=0), batch_size=1)[0]
        predictionIndex = np.where(prediction == np.amax(prediction))

        result_true = label_guide[labelIndex[0][0]]
        result_predict = label_guide[predictionIndex[0][0]]

        a = results_dict[result_true]
        a[result_predict] += 1

    # 3 groups, 3 bars per group
    data = np.zeros((3, 3))
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
            else:
                r = 2
            pred = results_dict[key]
            data[r][c] += pred[key2]

    X = np.arange(3)*1.2
    plt.bar(X + 0.0, data[0], color='lightsteelblue', width=0.25, label='normal')
    plt.bar(X + 0.25, data[1], color='forestgreen', width=0.25, label='pneumonia')
    plt.bar(X + 0.5, data[2], color='cyan', width=0.25, label='covid-pneumonia')

    plt.ylabel('Number of Occurances')
    plt.title('Predictions: label vs prediction')
    plt.xticks(X + 0.6, ('Normal', 'Pneumonia', 'Covid-Pneumonia'))
    plt.yticks(np.arange(0, 80, 5))

    plt.legend()

    plt.show()