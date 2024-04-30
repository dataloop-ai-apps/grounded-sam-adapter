import urllib.request
import torchvision
import logging
import pathlib
import torch
import time
import cv2
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from groundingdino.util.inference import Model, predict
from segment_anything import sam_model_registry, SamPredictor

import dtlpy as dl

logger = logging.getLogger('GroundedAdapter')


@dl.Package.decorators.module(description='Grounded SAM model adapter',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class GroundedSam(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'setting torch device: {device}')
        # PATHS
        grounded_dino_checkpoint_filepath = "artifacts/groundingdino_swint_ogc.pth"
        grounded_dino_config_filepath = pathlib.Path(__file__).parent / pathlib.Path(
            '../../utils/GroundingDINO_SwinT_OGC.py')
        grounded_dino_config_filepath = str(grounded_dino_config_filepath.resolve())
        grounded_dino_url = "https://storage.googleapis.com/model-mgmt-snapshots/grounded-dino/groundingdino_swint_ogc.pth"

        if not os.path.isfile(grounded_dino_checkpoint_filepath):
            os.makedirs(os.path.dirname(grounded_dino_checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(grounded_dino_url, grounded_dino_checkpoint_filepath)
        logger.info(f'loading weights grounded_dino_config_filepath: {grounded_dino_config_filepath}')
        logger.info(f'loading weights grounded_dino_checkpoint_filepath: {grounded_dino_checkpoint_filepath}')
        self.grounding_dino_model = Model(model_config_path=grounded_dino_config_filepath,
                                          model_checkpoint_path=grounded_dino_checkpoint_filepath,
                                          device=str(device))
        ############
        # Load SAM #
        ############
        sam_weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam/sam_vit_b_01ec64.pth'
        sam_checkpoint_filepath = self.configuration.get('sam_checkpoint_filepath', "artifacts/sam_vit_b_01ec64.pth")
        sam_checkpoint_url = self.configuration.get('sam_checkpoint_url', sam_weights_url)
        sam_model_type = self.configuration.get('sam_model_type', "vit_b")

        if not os.path.isfile(sam_checkpoint_filepath):
            os.makedirs(os.path.dirname(sam_checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(sam_checkpoint_url, sam_checkpoint_filepath)
        logger.info(f'loading weights sam_checkpoint_path: {sam_checkpoint_filepath}')
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_filepath)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def adjust_image_size(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        max_height = self.configuration.get('max_height', 640)
        max_width = self.configuration.get('max_width', 640)
        if height > width:
            if height > max_height:
                height, width = max_height, int(max_height / height * width)
        else:
            if width > max_width:
                height, width = int(max_width / width * height), max_width
        image = cv2.resize(image, (width, height))
        return image

    def create_annotations(self, detections, classes, original_shape, resized_shape, output_type):
        """
        Create Dataloop Annotations from the models' prediction output
        :param detections:
        :param original_shape: (x,y) shape of the original image
        :param resized_shape: (x,y) shape of the resized image
        :param classes: list of the model's classes
        :param output_type: output annotation type. binary, polygon

        :return: dl.AnnotationCollection
        """
        collection = dl.AnnotationCollection()
        for i_detection, detection in enumerate(detections):
            # if (detection["predicted_iou"] < predicted_iou_threshold
            #         or detection["stability_score"] < stability_score_threshold):

            # Get model results
            mask = detections.mask[i_detection]
            class_id = detections.class_id[i_detection]
            label = classes[class_id] if class_id is not None else 'NA'
            confidence = detections.confidence[i_detection]
            left, top, right, bottom = detections.xyxy[i_detection]

            # resize back to original shape
            mask = cv2.resize(mask.astype('uint8'), (original_shape[0], original_shape[1]))
            width_scale = original_shape[0] / resized_shape[0]
            height_scale = original_shape[1] / resized_shape[1]

            # Convert the coordinates back to the original image size
            left = int(left * width_scale)
            top = int(top * height_scale)
            right = int(right * width_scale)
            bottom = int(bottom * height_scale)

            model_info = {'name': self.model_entity.name,
                          'model_id': self.model_entity.id,
                          'confidence': float(confidence)}
            collection.add(dl.Box(left=left,
                                  top=top,
                                  bottom=bottom,
                                  right=right,
                                  label=label),
                           model_info=model_info)
            if output_type == 'binary':
                collection.add(dl.Segmentation(geo=mask,
                                               label=label),
                               model_info=model_info)
            else:
                collection.add(dl.Polygon.from_segmentation(mask=mask,
                                                            label=label),
                               model_info=model_info)
        return collection

    # Prompting SAM with detected boxes
    def segment_boxes(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(box=box,
                                                               multimask_output=True)
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def prepare_item_func(self, item):
        return item

    def predict(self, batch, **kwargs):
        # load image
        tic_total = time.time()
        pool = ThreadPoolExecutor(max_workers=16)
        batch_images = list(pool.map(self._item_to_image, batch))
        classes = self.configuration.get('classes', None)
        if classes is None:
            # get from item's recipe
            try:
                item: dl.Item = batch[0]
                labels = list(item.dataset.labels_flat_dict.keys())
                classes = {c: {'min_area': self.configuration.get('min_area', 0),
                               'max_area': self.configuration.get('max_area', np.inf)}
                           for c in labels}
            except Exception as e:
                logger.warning(f'Failed taking classes from recipe. Using default classes... Error was: {e}')
                classes = {'cat': {'min_area': 0,
                                   'max_area': np.inf},
                           'house': {'min_area': 0,
                                     'max_area': np.inf}
                           }

        if isinstance(classes, list):
            # classes list is used - need to deprecate
            classes = {c: {'min_area': self.configuration.get('min_area', 0),
                           'max_area': self.configuration.get('max_area', np.inf)}
                       for c in classes}
        # validation and add default area
        for c, val in classes.items():
            if 'min_area' not in val:
                val['min_area'] = 0
            if 'max_area' not in val:
                val['max_area'] = np.inf

        box_threshold = self.configuration.get('box_threshold', 0.2)
        text_threshold = self.configuration.get('text_threshold', 0.2)
        nms_threshold = self.configuration.get('nms_threshold', 0.8)
        output_type = self.configuration.get('output_type', 'polygon')  # box, binary
        with_nms = self.configuration.get('with_nms', True)
        save_results = kwargs.get('save_results')

        batch_annotations = list()
        for image in batch_images:
            # detect objects
            original_height = image.shape[0]
            original_width = image.shape[1]
            resized_image = self.adjust_image_size(image)
            resized_height = resized_image.shape[0]
            resized_width = resized_image.shape[1]

            # Inference models
            tic = time.time()
            classes_list = list(classes.keys())
            caption = ". ".join(classes_list)
            processed_image = Model.preprocess_image(image_bgr=resized_image).to(self.grounding_dino_model.device)
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model.model,
                image=processed_image,
                caption=caption,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                device=self.grounding_dino_model.device,
                remove_combined=True)
            source_h, source_w, _ = resized_image.shape
            detections = Model.post_process_result(
                source_h=source_h,
                source_w=source_w,
                boxes=boxes,
                logits=logits)
            class_id = Model.phrases2classes(phrases=phrases, classes=classes_list)
            detections.class_id = class_id

            # detections = self.grounding_dino_model.predict_with_classes(
            #     image=resized_image,
            #     classes=list(classes.keys()),
            #     box_threshold=box_threshold,
            #     text_threshold=text_threshold
            # )
            logger.info(f'Finished image Grounded DINO, time: {time.time() - tic:.2f}[s]')
            logger.info(f"Before NMS: {len(detections.xyxy)} boxes")
            # filter area
            to_keep_inds = list()
            for i_det in range(len(detections)):
                class_id = detections.class_id[i_det]
                area = detections.area[i_det]
                if detections.class_id[i_det] is None:
                    continue
                label = list(classes.keys())[class_id]
                if classes[label]['min_area'] < area < classes[label]['max_area']:
                    to_keep_inds.append(i_det)
                print(area, label)

            detections.xyxy = detections.xyxy[to_keep_inds]
            detections.confidence = detections.confidence[to_keep_inds]
            detections.class_id = detections.class_id[to_keep_inds]

            if with_nms is True:
                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    nms_threshold
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]
                logger.info(f"After NMS: {len(detections.xyxy)} boxes")
            # convert detections to masks
            tic = time.time()
            detections.mask = self.segment_boxes(
                image=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )
            logger.info(f'Finished SAM, time: {time.time() - tic:.2f}[s]')

            # annotate image with detections
            if save_results is True:
                import supervision as sv
                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()
                classes_list = list(classes.keys())
                labels = [classes_list[class_id] for class_id in detections.class_id]
                annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
                cv2.imwrite('detections.jpg', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

            if self._model_entity is not None:
                collection = self.create_annotations(detections=detections,
                                                     original_shape=(original_width, original_height),
                                                     resized_shape=(resized_width, resized_height),
                                                     classes=list(classes.keys()),
                                                     output_type=output_type)
                batch_annotations.append(collection)
        logger.info(f'Total detection time for {len(batch)} images: {time.time() - tic_total:.2f}[s]')
        return batch_annotations
