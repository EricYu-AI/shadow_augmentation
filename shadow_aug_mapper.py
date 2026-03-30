import copy
import numpy as np
from typing import Dict
import torch
import random

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper

from .shadow_augmentation import AugmentShadowOnFeature


aug_shadow = AugmentShadowOnFeature(
    hsigma=(100, 115), ssigma=(70, 120), vsigma=(10, 40),  # w/o color aug
    # hsigma=(100, 115), ssigma=(70, 120), vsigma=(40, 70),  # with color aug
)


class ShadowAugMapper(DatasetMapper):
    """ This callable which takes a dataset dict and map it into a format used
    by the model with additional shadow augmentation to input image in BGR
    format.
    """

    def __call__(self, dataset_dict: Dict) -> Dict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset
            format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        utils.check_image_size(dataset_dict, image)

        im = np.ascontiguousarray(image)

        # import cv2
        # cv2.imshow("original", im)

        if random.uniform(0, 1) < 0.50:
            aug_coords = np.random.randint(10, high=1269, size=(1, 2))
            for sl in aug_coords:
                aug_shadow.apply_random_shadow(im, sl)

        used_locs = []
        feat_centers = []
        for inst in dataset_dict["annotations"]:
            x, y, w, h = inst['bbox']
            feat_centers.append((int(x + w/2), int(y + h/2)))
        feat_centers = np.array(feat_centers)
        np.random.shuffle(feat_centers)
        for sl in feat_centers:
            no_overlap = True
            for loc in used_locs:
                dist = np.linalg.norm(sl - loc)
                if dist < aug_shadow.shadow_size:
                    no_overlap = False
            if random.uniform(0, 1) < 0.5 and no_overlap:
                aug_shadow.apply_random_shadow(im, sl)
                used_locs.append(sl)

        aug_input = T.AugInput(im)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


__all__ = ["ShadowAugMapper"]
