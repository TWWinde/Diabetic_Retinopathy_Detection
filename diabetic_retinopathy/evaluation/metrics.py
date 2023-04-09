import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf



def greater_than_0_5(x):
    return tf.where(x > 0.5, 1, 0)


alpha = 0.25
gamma = 1.0


class DrawConfusionMatrix:
    def __init__(self, labels_name, normalize=True):
        """normalize：if set number to percentage"""
        self.normalize = normalize
        self.labels_name = labels_name
        self.num_classes = len(labels_name)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype="float32")
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype="float32")

    def update(self, labels, predicts):
        """write labels predicts as one dime vector"""
        for label, predict in zip(labels, predicts):
            self.matrix[label, predict] += 1

        return self.matrix

    def getMatrix(self, normalize=True):
        """
        if normalize=True，percentage，
        if normalize=False，number
        Returns a matrix with number or percentage
        """
        if normalize:
            per_sum = self.matrix.sum(axis=1)  # row-sum
            for i in range(self.num_classes):
                self.matrix[i] = (self.matrix[i] / per_sum[i])
            self.matrix = np.around(self.matrix, 2)
            self.matrix[np.isnan(self.matrix)] = 0
        return self.matrix

    def drawMatrix(self):
        self.matrix = self.getMatrix(self.normalize)
        plt.imshow(self.matrix, cmap=plt.cm.Blues)
        plt.title("Normalized confusion matrix")  # title
        plt.xlabel("Predict label")
        plt.ylabel("Truth label")
        plt.yticks(range(self.num_classes), self.labels_name)
        plt.xticks(range(self.num_classes), self.labels_name, rotation=0)

        for x in range(self.num_classes):
            for y in range(self.num_classes):
                value = float(format('%.2f' % self.matrix[y, x]))
                plt.text(x, y, value, verticalalignment='center', horizontalalignment='center')

        plt.tight_layout()

        plt.colorbar()
        plt.savefig('./ConfusionMatrix.png', bbox_inches='tight')
        plt.show()


labels_name = ['NRDR', 'RDR']
drawconfusionmatrix = DrawConfusionMatrix(labels_name=labels_name)


def confusionmatrix(model, ds_test):
    for images, labels in ds_test:
        prediction = model(images, training=False)
        prediction = greater_than_0_5(prediction)
        prediction = prediction.numpy()
        predict_np = np.squeeze(prediction)
        # print(prediction)
        # predict_np = np.argmax(prediction.numpy(), axis=1)
        labels_np = labels.numpy()
        matrix = drawconfusionmatrix.update(labels_np, predict_np)

    drawconfusionmatrix.drawMatrix()
    confusion_mat = drawconfusionmatrix.getMatrix()
    print(confusion_mat)

    Precision = matrix[1, 1] / (matrix[1, 1] + matrix[0, 1])
    print('Precision = ', Precision)
    Sensitivity = matrix[1, 1] / (matrix[1, 1] + matrix[1, 0])
    print('Sensitivity = ', Sensitivity)
    Specificity = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    print('Specificity = ', Specificity)

    '''ROC AUC'''


def ROC(model, ds_test):
    for images, labels in ds_test:
        y_true = np.array(labels)
        y_score = np.array(model(images, training=False))
    # y_score = np.delete(y_score, 0, axis=1)
    print(y_score)
    print(y_score.shape)
    # Calculate the AUC score
    auc = roc_auc_score(y_true, y_score)
    print("AUC: ", auc)
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve of VGGNet (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
