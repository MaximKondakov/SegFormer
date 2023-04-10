import cv2 as cv
import numpy as np
from skimage.measure import regionprops
from skimage.morphology import label
from config_category_label_and_color import category_ids, color_map


def prob_and_name_in_bbox(image, classes_in_crop_segmentation_map, probability_in_crop_segmentation, coordinates_roi):
    """
    function uses get_name_bbox() and  get_contour() if confidence score above threshold.
    threshold_confidence = const, for avoid anomaly case predict of the model or lower value
    :param image: full image with segmentation
    :param classes_in_crop_segmentation_map: tensor [height, width] after argmax
    :param probability_in_crop_segmentation: tensor [num_class, height, width] after softmax
    :param coordinates_roi:
    :return: image with contour and name with confidence score
    """

    threshold_confidence = 0.6
    # search regions on the segmentation map
    connected_regions = label(classes_in_crop_segmentation_map)  # Label connected regions
    regions_property = regionprops(connected_regions)
    number_regions = len(regions_property)
    # loop for each region on the image
    contour_list_regions = []
    label_xy_list = []
    conf_i_class_list = []
    for i_region in range(number_regions):
        min_row, min_col, max_row, max_col = regions_property[i_region].bbox
        probability_segment_region = probability_in_crop_segmentation[:, min_row:max_row, min_col:max_col]
        probability_in_i_region_numpy = probability_segment_region.detach().numpy()

        classes_in_i_region = classes_in_crop_segmentation_map[min_row:max_row, min_col:max_col]
        classes_in_i_region = classes_in_i_region.detach().numpy()

        unique_classes_in_i_region = np.unique(classes_in_i_region)
        unique_classes_in_i_region = unique_classes_in_i_region[
            unique_classes_in_i_region != 0]  # first id is the background
        for i_class in unique_classes_in_i_region:
            probability_in_i_region_numpy[i_class] = np.where(
                ((probability_in_i_region_numpy[i_class] != 0) & (classes_in_i_region != 0)),
                probability_in_i_region_numpy[i_class], np.nan)
            conf_i_class = np.nanmean(probability_in_i_region_numpy[i_class])

            # if confidence i_class more than threshold -> draw contour, probability and name
            if conf_i_class >= threshold_confidence:
                contour_region = get_contour(image, classes_in_i_region, regions_property[i_region], coordinates_roi)
                contour_list_regions.append(contour_region)
                # get_rect_bbox(image, regions_property[i_region], coordinates_roi)
                label_x, label_y = get_name_bbox(image, i_class, regions_property[i_region], conf_i_class, coordinates_roi)
                label_xy_list.append([label_x, label_y])

    return image, contour_list_regions, label_xy_list, conf_i_class_list


def get_name_bbox(image, i_class, region_property, conf_i_class, coordinates_roi):
    # parameters
    image_height, image_weight = image.shape[0], image.shape[1]
    my_color_map = color_map()
    x, y, w, h = coordinates_roi
    font = cv.FONT_HERSHEY_SIMPLEX  # font
    line_type = cv.LINE_AA
    font_scale = max(image_weight, image_height) * 0.0007
    thickness = int(max(image_weight, image_height) * 0.002)
    dict_category_ids = category_ids()

    for index, name_class in enumerate(dict_category_ids):
        if index == i_class:
            confidence_score_str = str(round(conf_i_class, 2))
            text = f"{name_class}:{confidence_score_str}"
            (label_width, label_height), label_size = cv.getTextSize(text, font, font_scale, thickness)
            label_x = region_property.bbox[1] + x
            label_y = region_property.bbox[0] + y + label_height
            location = (label_x, label_y)

            color_rect = my_color_map[i_class]  # color text
            cv.rectangle(image, (region_property.bbox[1] + x, region_property.bbox[0]+y),
                         (region_property.bbox[1] + label_width + x,  region_property.bbox[0]+y+label_height),
                         color_rect, -1)

            # color = my_color_map[i_class]  # color text
            color_text = (255, 255, 255)
            cv.putText(image, text, location, font, font_scale, color_text, thickness, line_type)

    return label_x, label_y


def get_rect_bbox(image, region_property, coordinates_roi):
    x, y, w, h = coordinates_roi
    color = (255, 0, 0)
    cv.rectangle(image, (region_property.bbox[1] + x, region_property.bbox[0] + y),
                 (region_property.bbox[3] + x, region_property.bbox[2] + y),
                 color, 1)


def get_contour(image, classes_in_i_region, region_property, coordinates_roi):
    x, y, w, h = coordinates_roi
    x = region_property.bbox[1] + x  # total shift by x_axis
    y = region_property.bbox[0] + y  # total shift by y_axis

    contours, hierarchy = cv.findContours(classes_in_i_region.astype(np.uint8),
                                          cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(x, y))
    contour_region = contours[0]
    # eps = 0.001  # the number of points starts to saturate
    # peri = cv.arcLength(contours[0], True)
    # approx = cv.approxPolyDP(contours[0], eps * peri, True)
    # print('_'*50)
    # print(f"number points original= {len(contours[0])}")
    # print(f"number points approx= {len(approx)}")
    # cv.drawContours(image, [approx], -1, (255, 0, 0), 3)
    cv.drawContours(image, contours, -1, (255, 0, 0), 3)

    return contour_region


def get_img_seg(image, segmentation):
    """
    function for colorize mask and merge mask + image
    """
    color_seg = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    my_color_map = color_map()
    for id_color, color in my_color_map.items():
        color_seg[segmentation == id_color] = my_color_map[id_color]
    image = np.array(image) * 0.5 + color_seg * 0.5
    image = image.astype(np.uint8)

    return image
