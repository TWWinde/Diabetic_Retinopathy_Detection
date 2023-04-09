import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def Grad_CAM(model, img_path, last_conv_layer_name):
    image = cv2.imread(img_path)
    image = np.array(image) / 255
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    img_array = np.expand_dims(image, axis=0)
    # Remove last layer's softmax
    model.layers[-1].activation = None
    # Print what the top predicted class is
    predictions = model(img_array, training=False)
    print("Predicted:", np.argmax(predictions.numpy(), axis=-1))

    # Generate class activation heatmap
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions

        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        # grads.shape(1, 10, 10, 2048)
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        # last_conv_layer_output.shape  =(10, 10, 2048)
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        # tf.squeeze 去除1的维度,(10, 10)
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1

        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()
    # process the heatmap and superposition
    heatmap = cv2.resize(heatmap, (256, 256))
    r = heatmap
    heatmap = cv2.merge([r, r, r])
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    b, g, r = cv2.split(image)
    image = cv2.merge([r, g, b])
    cam = heatmap * 0.6 + np.float32(image)
    cam = cam / np.max(cam)
    plt.imshow(cam)
    plt.show()
    cv2.imwrite("Heatmap.jpg", cam * 255)
