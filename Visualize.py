import tensorflow as tf
import matplotlib.pyplot as plt
import os

from Grad_CAM import *
import argparse

def check_filePath(path: str):
    """
    Check if the path exists. If this is not the case, an error will be raised.
    @param path: path to be checked
    @return path if it exists
    """
    
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError("The path to the image does not exists.")
    
    return path 


def load_img(path, adapt_to_preprocess=False):
    """
    Load the image from a file
    
    @param path: Path to the image
    @param adapt_to_preprocess: true -> img rescaled in the range of [0, 255], resized (224,224) 
                                        and batch dim is added
                                false -> default float image (range: [0, 1]) 
    """
    
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    if adapt_to_preprocess:
        img = img*255    
        img = tf.image.resize(img, (224, 224))

        # Add batch dim
        img = tf.expand_dims(img, axis=0)

    return img


def main():

    #
    # Set up argparse
    #

    parser = argparse.ArgumentParser(description="Visualize Grad-CAM")
    parser.add_argument("--path", help="Set the path to the image to which the Grad-CAM algorithm shall be applied.", type=check_filePath, required=True)
    args = parser.parse_args()

    #
    # Set up model
    #
    img_path = args.path

    vgg = tf.keras.applications.VGG19(include_top=True, 
                                           weights='imagenet', 
                                           classifier_activation=None)
    vgg.trainable = False
    
    last_conv_layer_name = "block5_conv4"
    model = tf.keras.Model(
        vgg.inputs, [vgg.get_layer(last_conv_layer_name).output, vgg.output]
    )

    #
    # Grad-CAM
    #

    # Create Grad-CAM
    img = load_img(img_path, adapt_to_preprocess=True)
    
    img = tf.keras.applications.vgg19.preprocess_input(img)
    grad_cam, grad_class_wrt_img, _ = get_grad_cam(img, model, guided=True)

    # Overlay it with input image
    img = load_img(img_path, adapt_to_preprocess=False)
    img = img.numpy()

    overlay = create_overlay(img, grad_cam)
    
    #
    # Guided Grad-CAM
    #
    guided_grad_cam = get_guided_grad_cam(grad_cam, grad_class_wrt_img, img)


    #
    # Plot
    #

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].imshow(img)
    axs[0].set_title("Input")

    axs[1].imshow(overlay)
    axs[1].set_title("Grad-CAM")

    axs[2].imshow(guided_grad_cam)
    axs[2].set_title("Guided Grad-CAM")

    for ax in axs:
        ax.axis("off")
    
    plt.tight_layout()
    file_name = os.path.basename(img_path)
    plt.savefig(f"./results/{file_name}", dpi=300)
    plt.show()
 

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")