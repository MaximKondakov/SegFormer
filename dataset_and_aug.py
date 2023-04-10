import os
import albumentations as aug
import cv2
from torch.utils.data import Dataset
from transformers import SegformerImageProcessor


class ImageSegmentationDataset(Dataset):
    """
    Semantic segmentation maps stored in the png format
    Dataset stores the samples and their corresponding labels
    """
    def __init__(self, root_dir, feature_extractor, transforms=None, train=True):
        """
        :param root_dir: Root directory of the dataset containing the images + annotations
        :param feature_extractor: (SegformerImageProcessor): feature extractor to prepare images + segmentation maps
        :param transforms: albumentation transforms for robust training
        :param train: Whether to load "training" or "validation" images + annotations
        """

        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.train = train
        self.transforms = transforms

        sub_path = "train" if self.train else "val"

        # paths to the folder images and masks
        self.img_dir = os.path.join(self.root_dir, sub_path, "images")
        self.ann_dir = os.path.join(self.root_dir, sub_path, "masks")

        # read images
        image_file_names = []
        for root, dirs, files in os.walk(self.img_dir):
            image_file_names.extend(files)
        self.images = sorted(image_file_names)

        # read annotations
        annotation_file_names = []
        for root, dirs, files in os.walk(self.ann_dir):
            annotation_file_names.extend(files)
        self.annotations = sorted(annotation_file_names)

        assert len(self.images) == len(self.annotations), "There must be as many images as there are segmentation maps"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        segmentation_map = cv2.imread(os.path.join(self.ann_dir, self.annotations[idx]))
        segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_BGR2GRAY)

        x, y, w, h = self.find_relevant_region(image)
        image = image[y:y + h, x:x + w]
        segmentation_map = segmentation_map[y:y + h, x:x + w]

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=segmentation_map)
            # randomly crop + pad both image and segmentation map to same size
            encoded_inputs = self.feature_extractor(augmented['image'], augmented['mask'], return_tensors="pt")
        else:
            encoded_inputs = self.feature_extractor(image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension
            # encoded_inputs[k] = np.squeeze(encoded_inputs[k])
        return encoded_inputs

    @staticmethod
    def find_relevant_region(image):
        """
        image preprocessing
        otsu's method  - threshold that separate pixels into two classes, foreground and background
        coordinates -  cv2.boundingRect
        :param image: image
        :return: coordinates relevant region (cargo)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (25, 25), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Perform morph operations, first open to remove noise, then close to combine
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

        # Find enclosing boundingbox and crop ROI
        coordinates = cv2.findNonZero(close)
        x, y, w, h = cv2.boundingRect(coordinates)

        return x, y, w, h


def main(data_dir):
    """
    transforms validation and train dataset
    perform:
        1) transforms - with Albumentation,
        2) find and crop relevant region - using by using otsu's method
        3) normalize - using feature extractor from huggingface
    :param data_dir: path for images and masks
    :return: set below steps for image preprocessing on fly while train a model
    """

    prob_p = 0.4
    transform = aug.Compose([
        aug.Flip(p=prob_p),
        aug.RandomRotate90(p=prob_p),
        aug.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=prob_p),
        aug.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=prob_p),
        aug.GaussNoise(p=prob_p),
        aug.OneOf([
            aug.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=prob_p),
            aug.ElasticTransform(p=prob_p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            aug.GridDistortion(p=prob_p),
            aug.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=prob_p),
        ], p=prob_p)
    ])
    # create dataset
    data_dir = data_dir
    feature_extractor = SegformerImageProcessor(align=False, do_reduce_labels=False)

    train_dataset = ImageSegmentationDataset(root_dir=data_dir,
                                             feature_extractor=feature_extractor,
                                             transforms=transform)
    valid_dataset = ImageSegmentationDataset(root_dir=data_dir,
                                             feature_extractor=feature_extractor,
                                             transforms=None, train=False)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(valid_dataset))
    return train_dataset, valid_dataset


if __name__ == '__main__':
    main(data_dir=None)
