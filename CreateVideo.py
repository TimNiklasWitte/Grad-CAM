import tensorflow as tf
import cv2
import os
import argparse

from Grad_CAM import *


def check_filePath(path: str):
    """
    Check if the path exists. If this is not the case, an error will be raised.
    @param path: path to be checked
    @return path if it exists
    """
    
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError("The path to the video does not exists.")
    
    return path 

def preprocess(frame):
    """
    Preprocess the image in a way that it can be processed by the VGG19 network.
    In other words, it is resized to (224, 224), a batch dimension is added and 
    the preprocess_input of this network is called.
    
    @param frame: image to be preprocessed
    @return preprocessed image aka frame
    """
    frame = tf.image.resize(frame, (224, 224))
    frame = tf.expand_dims(frame, axis=0)
    frame = tf.keras.applications.vgg19.preprocess_input(frame)

    return frame

def convert_colors_to_viridis(img):
    """
    Convert the img (grey scaled) into colors (RGB) based on the viridis color map.

    @param img: Image in the range of [0, 1] (numpy array, float32)
    @return converted image
    """

    #
    # Bring to range [0, 255] -> use values as index for color values of color map
    #
    img = np.uint8(255 * img)

    viridis = plt.colormaps["viridis"]
    viridis_colors = viridis(np.arange(256))[:, :3]
    viridis_heatmap = viridis_colors[img]

    return viridis_heatmap

def main():

    #
    # Set up argparse
    #

    parser = argparse.ArgumentParser(description="Visualize Grad-CAM in a video. Use q to quit.")
    parser.add_argument("--path", help="Set the path to the input video.", type=check_filePath, required=True)
    parser.add_argument("--out", help="Set name of the resulting output .MP4 file.", required=True)
    args = parser.parse_args()

    input_video_path = args.path 
    output_video_name = args.out

    #
    # Set up model
    #

    vgg = tf.keras.applications.VGG19(include_top=True, 
                                           weights='imagenet', 
                                           classifier_activation=None)
    vgg.trainable = False
    
    last_conv_layer_name = "block5_conv4"
    model = tf.keras.Model(
        vgg.inputs, [vgg.get_layer(last_conv_layer_name).output, vgg.output]
    )


    #
    # Config video stream reader and writer
    #

    scale_ratio = 1/2

    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(f'./{output_video_name}.mp4', fourcc, 20.0, (1920, 540))

    while cap.isOpened():
        ret, frame = cap.read()

        # Frame is not read correctly -> end
        if not ret:
            break

        #
        # Frame shall be always in the range of [0, 1]
        #

        # only here in the range of [0, 255]
        frame = frame.astype(np.float32)
        frame_tf = preprocess(frame)

        frame = frame/255

        grad_cam, grad_class_wrt_img, preds = get_grad_cam(frame_tf, model, guided=True)

        overlay = create_overlay(frame, grad_cam)
    

        #
        # Rescale
        #
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

        scaled_width = int(width*scale_ratio)
        scaled_height = int(height*scale_ratio)

        overlay = cv2.resize(overlay, (scaled_width, scaled_height)).astype(np.float32)

        #
        # Add text to overlay
        # 

        font = cv2.FONT_HERSHEY_SIMPLEX 
        fontScale = 1
        color = (1, 0, 0) # Blue color in BGR 

        # Line thickness
        thickness = 2 # px 

        # Extract best prediction
        preds = tf.nn.softmax(preds)
        decoded_preds = tf.keras.applications.vgg19.decode_predictions(preds.numpy())[0]
        decoded_preds = decoded_preds[0] # class with highest value
        _ , class_name, prob = decoded_preds

        # Prediction: prob
        overlay = cv2.putText(overlay, f"{class_name}: {prob:1.3f}", (0, scaled_height - 10), font,  
                        fontScale, color, thickness, cv2.LINE_AA) 

        # Frame title: Grad-CAM
        overlay = cv2.putText(overlay, "Grad-CAM", (0, 25), font,  
                        fontScale, color, thickness, cv2.LINE_AA) 
        
        #
        # Get Guided Grad-CAM
        #
        guided_grad_cam = get_guided_grad_cam(grad_cam, grad_class_wrt_img, overlay)
        guided_grad_cam = convert_colors_to_viridis(guided_grad_cam)

        # Frame title: Guided Grad-CAM
        guided_grad_cam = cv2.putText(guided_grad_cam, "Guided Grad-CAM", (0, 25), font,  
                        fontScale, color, thickness, cv2.LINE_AA)
        
        #
        # Put Grad-CAM and Guided Grad-CAM together
        #
        display = np.zeros(shape=(scaled_height, scaled_width*2, 3), dtype=np.float32)
        display[:scaled_height, scaled_width:, :]  = guided_grad_cam
        display[:scaled_height, :scaled_width, :]  = overlay

        display = (display*255).astype(np.uint8)
        cv2.imshow('Grad-CAM', display)

        out.write(display)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")