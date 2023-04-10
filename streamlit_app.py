import streamlit as st
import torch
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import os
from dataset_and_aug import ImageSegmentationDataset
from torch import nn
from model_plot import get_img_seg, prob_and_name_in_bbox
import numpy as np
import cv2


def build_model():
    """
    Initialize/load model weights
    :return: pretrained model
    """
    input_path = os.getcwd()
    best_model_name = 'continue/checkpoint-4400'

    # load model from checkpoint
    model = SegformerForSemanticSegmentation.from_pretrained(
        os.path.join(input_path, best_model_name))

    return model


def predict(image):
    """
    perform  initialize model -> find_relevant_region -> predict -> merge img and seg -> draw helpful info
    :param image: input image
    :return: result segmented image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = model.to(device)
    coordinates_roi = x, y, w, h = ImageSegmentationDataset.find_relevant_region(image)  # find cargo

    # crop image for better performance
    crop_image = image[y:y + h, x:x + w]

    # select feature extractor
    feature_extractor_inference = SegformerImageProcessor(do_random_crop=False, do_pad=False, do_reduce_labels=False)
    pixel_values = feature_extractor_inference(crop_image, return_tensors="pt").pixel_values.to(device)
    model.eval()
    outputs = model(pixel_values=pixel_values)  # logits are of shape (batch_size, num_labels, height/4, width/4)
    logits = outputs.logits.cpu()

    # First, rescale logits to original image size
    upsampled_logits = nn.functional.interpolate(logits,
                                                 size=crop_image.shape[::][:-1], # (height, width)
                                                 mode='bilinear',
                                                 align_corners=False)

    # Second, apply argmax on the class dimension
    classes_in_crop_segmentation_map = upsampled_logits.argmax(dim=1)[0]
    probability_in_crop_segmentation = upsampled_logits.softmax(dim=1)[0]  # probability tensor [channel, width, height]

    original_segmentation_size = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    original_segmentation_size[y:y + h, x:x + w][:] = classes_in_crop_segmentation_map[:, :]

    segmentation_and_image = get_img_seg(image, original_segmentation_size)  # second image+ segmentation

    res_img, contours, label_xy_list, conf_i_class_list = prob_and_name_in_bbox(segmentation_and_image, classes_in_crop_segmentation_map,
                                    probability_in_crop_segmentation, coordinates_roi)
    print(f'number are polygons = {len(contours)}')
    print(f'number points = {len(contours[0])}')
    print(contours[0][-1][0])
    return res_img


def convert_image(img):
    """
    Save final segmented image for download_button
    :param img: image
    :return: bytes image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.imencode('.jpeg', img)[1].tobytes()
    return img


def fix_image(image, name_image):
    """
    Main page
    :param image: input image
    :param name_image: for download_button
    :return:
    """
    col1.write("Original Image :camera:")
    col1.image(image)
    fixed = predict(image)
    col2.write("Fixed Image :sparkles:")
    col2.image(fixed)
    st.sidebar.markdown("\n")
    st.sidebar.download_button(label="Download fixed image",
                               data=convert_image(fixed),
                               file_name=f"{name_image}",
                               mime="image/jpeg")


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Segformer segmentation Maksim")
    st.write("# Find items from your image")
    st.write(
        "Test page to view result model"
    )
    st.sidebar.write("## Upload and download :gear:")

    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader(label="Upload an image",
                                         type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        file_bytes = np.asarray(bytearray(my_upload.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        name_image = my_upload.name
        fix_image(opencv_image, name_image)
    else:
        default_img = "/home/maksim/PycharmProjects/pythonProject/data/cargo_segform_train=0.8val=0.2/train/images" \
                      "/81_.jpeg"
        name_image = '81_'
        opencv_image = cv2.imread(default_img)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        fix_image(opencv_image, name_image)
    # streamlit run segformer_model/streamlit_app.py
