import tensorflow as tf
import cv2

import numpy as np
import cv2

import matplotlib.pyplot as plt


def get_grad_cam(img, model, guided=False):
    """
    Create a Grad-CAM heatmap

    @param img: Preprocessed img by tf.keras.applications.vgg19.preprocess_input
    @param model: TensorFlow model
    @return heatmap in the range of [0, 1] (numpy array, float32)
    """

    img_var = tf.Variable(img)
    with tf.GradientTape(persistent=True) as tape:
        last_conv_layer_output, preds = model(img_var)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grad_class_wrt_conv = tape.gradient(class_channel, last_conv_layer_output)
       

    # Remove batch dim
    grad_class_wrt_conv = grad_class_wrt_conv[0]
    last_conv_layer_output = last_conv_layer_output[0]

    # Weighting
    activation_map_weights = tf.reduce_mean(grad_class_wrt_conv, axis=(0, 1))

    heatmap = tf.reduce_sum(last_conv_layer_output * activation_map_weights, axis=-1)
    heatmap = tf.nn.relu(heatmap)

    # Normalize between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    if guided:
        grad_class_wrt_img = tape.gradient(class_channel, img_var)

        grad_class_wrt_img = grad_class_wrt_img[0] # remove batch dim

        # Normalize between 0 and 1
        grad_class_wrt_img = tf.maximum(grad_class_wrt_img, 0) / tf.math.reduce_max(grad_class_wrt_img)
        
        # Average across color channels
        grad_class_wrt_img = tf.reduce_mean(grad_class_wrt_img, axis=-1)
        grad_class_wrt_img = grad_class_wrt_img.numpy()

        return heatmap, grad_class_wrt_img, preds

    return heatmap, preds


def create_overlay(img, grad_cam, alpha=0.4):
    """
    Overlay the image with its corresponding Grad-CAM (get_grad_cam)

    @param img: Image in the range of [0, 1] (numpy array, float32)
    @param model: TensorFlow model
    @return overlay in the range of [0, 1] (numpy array, float32)
    """

    #
    # Bring to range [0, 255] -> use values as index for color values of color map
    #
    grad_cam = np.uint8(255 * grad_cam)

    jet = plt.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_colors[:50, :] = 0 # remove blue touch in overlayed img

    jet_grad_cam= jet_colors[grad_cam]

    # Resize
    jet_grad_cam = cv2.resize(jet_grad_cam, (img.shape[1], img.shape[0])).astype(np.float32)
    
    # Overlay img and jet_grad_cam
    overlay = cv2.addWeighted(img, 1, jet_grad_cam, alpha, 0)
    overlay = np.minimum(overlay, 1)
    
    return overlay

def get_guided_grad_cam(grad_cam, grad_class_wrt_img, img):

    """
    Overlay the image with its corresponding Grad-CAM (get_grad_cam)

    @param img: Image in the range of [0, 1] (numpy array, float32)
    @param model: TensorFlow model
    @return overlay in the range of [0, 1] (numpy array, float32)
    """

    # Same size as input image (network!)
    grad_cam = cv2.resize(grad_cam, (224, 224)).astype(np.float32)

    # Element-wise multiplication
    guided_grad_cam = grad_cam * grad_class_wrt_img

    # Now same as original image (before preprocessing)
    guided_grad_cam = cv2.resize(guided_grad_cam, (img.shape[1], img.shape[0])).astype(np.float32)

    return guided_grad_cam

