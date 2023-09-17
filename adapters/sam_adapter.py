import logging
import os
import urllib
from random import randint
import typing
import dtlpy as dl
import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

"""


"""
logger = logging.getLogger('SamAdapter')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WIDTH = MAX_HEIGHT = 640


@dl.Package.decorators.module(description='Model Adapter for Segment Anything',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class SegmentAnythingAdapter(dl.BaseModelAdapter):

    def load(self, local_path, **kwargs):
        logger.info(f'loading model weights. device: {DEVICE}')
        checkpoint_filepath = self.configuration.get('checkpoint_filepath', '')
        checkpoint_url = self.configuration.get('checkpoint_url')
        model_type = self.configuration.get('model_type')

        # SAM Options
        box_nms_thresh = self.configuration.get('box_nms_thresh', 0.7)
        points_per_side = self.configuration.get('points_per_side', 32)
        crop_n_layers = self.configuration.get('crop_n_layers', 0)
        crop_n_points_downscale_factor = self.configuration.get('crop_n_points_downscale_factor', 1)
        points_per_batch = self.configuration.get('points_per_batch', 64)
        pred_iou_thresh = self.configuration.get('pred_iou_thresh', 0.88)
        stability_score_thresh = self.configuration.get('stability_score_thresh', 0.95)
        stability_score_offset = self.configuration.get('stability_score_offset', 1.0)
        crop_nms_thresh = self.configuration.get('crop_nms_thresh', 0.7)
        crop_overlap_ratio = self.configuration.get('crop_overlap_ratio', 512 / 1500)
        min_mask_region_area = self.configuration.get('min_mask_region_area', 0)
        ################

        if not os.path.isfile(checkpoint_filepath):
            checkpoint_filepath = os.path.join('/tmp/app', checkpoint_filepath)
        if not os.path.isfile(checkpoint_filepath):
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint_filepath)
        logger.info(f'loading model weights from {checkpoint_filepath}')
        sam = sam_model_registry[model_type](checkpoint=checkpoint_filepath).to(DEVICE)
        self.model = SamAutomaticMaskGenerator(model=sam,
                                               box_nms_thresh=box_nms_thresh,
                                               points_per_side=points_per_side,
                                               crop_n_layers=crop_n_layers,
                                               crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                               points_per_batch=points_per_batch,
                                               crop_overlap_ratio=crop_overlap_ratio,
                                               crop_nms_thresh=crop_nms_thresh,
                                               stability_score_offset=stability_score_offset,
                                               stability_score_thresh=stability_score_thresh,
                                               pred_iou_thresh=pred_iou_thresh,
                                               min_mask_region_area=min_mask_region_area
                                               )

    def adjust_image_size(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height > width:
            if height > MAX_HEIGHT:
                height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
        else:
            if width > MAX_WIDTH:
                height, width = int(MAX_WIDTH / width * height), MAX_WIDTH
        image = cv2.resize(image, (width, height))
        return image

    def draw_masks(self, image: np.ndarray, masks: typing.List[np.ndarray], alpha: float = 0.7) -> np.ndarray:
        import cv2
        for mask in masks:
            color = [randint(127, 255) for _ in range(3)]

            # draw mask overlay
            colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
            colored_mask = np.moveaxis(colored_mask, 0, -1)
            masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
            image_overlay = masked.filled()
            image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

            # draw contour
            contours, _ = cv2.findContours(
                np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        return image

    def predict(self, batch, **kwargs):
        stability_score_threshold = 0.85
        predicted_iou_threshold = 0.85
        output_type = self.configuration.get('output_type', 'binary')
        batch_annotations = list()
        for image in batch:
            reshaped_image = self.adjust_image_size(image=image)
            masks = self.model.generate(reshaped_image)
            collection = dl.AnnotationCollection()
            for i_detection, detection in enumerate(masks):
                # if (detection["predicted_iou"] < predicted_iou_threshold
                #         or detection["stability_score"] < stability_score_threshold):
                mask = cv2.resize(detection['segmentation'].astype('uint8'), (image.shape[1], image.shape[0]))
                model_info = {'name': self.model_entity.name,
                              'model_id': self.model_entity.id,
                              'confidence': float(detection['stability_score'])}
                if output_type == 'binary':
                    collection.add(dl.Segmentation(geo=mask,
                                                   label=f'poly{i_detection}'),
                                   model_info=model_info)
                elif output_type == 'box':
                    poly = dl.Polygon.from_segmentation(mask=mask,
                                                        label=f'mask-{i_detection}',
                                                        max_instances=1)
                    collection.add(dl.Box(top=poly.top,
                                          left=poly.left,
                                          bottom=poly.bottom,
                                          right=poly.right,
                                          label=f'box-{i_detection}'),
                                   model_info=model_info)

                else:
                    collection.add(dl.Polygon.from_segmentation(mask=mask,
                                                                label=f'poly-{i_detection}'),
                                   model_info=model_info)
                batch_annotations.append(collection)

        return batch_annotations


def test():
    self = SegmentAnythingAdapter()
    m = dl.models.get(None, '648956679a02a7fac095eedc')
    self.load_from_model(model_entity=m)
    dl.setenv('prod')
    item = dl.items.get(item_id='64871ee6e98a8633ddfe91ed')
    self.predict_items(items=[item])
