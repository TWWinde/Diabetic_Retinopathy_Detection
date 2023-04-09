import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def Dimensionality_Reduction(model, ds_test):
    second_last_layer_name = 'dense'
    dr_model = tf.keras.models.Model([model.inputs], [model.get_layer(second_last_layer_name).output])
    outputs = []
    labels = []
    predictions = []
    count = 0
    for image, label in ds_test:
        image = np.array(image)
        # image = np.expand_dims(image, axis=0)
        output = dr_model(image, training=False).numpy()
        prediction = np.argmax(model(image, training=False).numpy(), axis=-1)
        label = label.numpy()
        if label == prediction:
            count += 1
        outputs.append(output)
        labels.append(label)
        predictions.append(prediction)
    outputs = np.concatenate(outputs)
    print(count)
    '''T-SEN'''
    tsne = TSNE(n_components=2, learning_rate=200, metric='cosine', n_jobs=-1)
    tsne.fit_transform(outputs)
    outs_2d = np.array(tsne.embedding_)
    a = []
    b = []
    for i in range(len(outs_2d)):
        if labels[i] == 1:
            a.append([outs_2d[i, 0], outs_2d[i, 1]])
        else:
            b.append([outs_2d[i, 0], outs_2d[i, 1]])
    a = np.asarray(a)
    b = np.asarray(b)
    print(a, b)
    plt.plot(a[:, 0], a[:, 1], '.', label='RDR', color='green')
    plt.plot(b[:, 0], b[:, 1], '.', label='NRDR', color='blue')
    plt.title('Dimensionality reduction visualization by tSNE,test data')
    plt.legend(loc="lower right")
    plt.show()
