from PIL import Image
import urllib.request
import collections
import subprocess
import threading
import pydantic
import datetime
import logging
import base64
import torch
import json
import time
import tqdm
import os
import cv2
import dtlpy as dl
import numpy as np

from sam2.build_sam import build_sam2

from adapters.global_sam_adapter.sam2_handler import DataloopSamPredictor

logger = logging.getLogger('[SAM]')
logger.setLevel('INFO')


class CachedItem(pydantic.BaseModel):
    image_embed: torch.Tensor
    timestamp: datetime.datetime
    high_res_feats: tuple
    orig_hw: tuple

    class Config:
        arbitrary_types_allowed = True


class Runner(dl.BaseServiceRunner):
    def __init__(self):
        """
        Init package attributes here

        :return:
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'GPU available: {torch.cuda.is_available()}')

        model_cfg = "sam2_hiera_s.yaml"
        weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_small.pt'
        weights_filepath = 'artifacts/sam2_hiera_small.pt'
        if not os.path.isfile(weights_filepath):
            os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_filepath)

        sam2_model = build_sam2(model_cfg, weights_filepath, device=device)
        self.predictor = DataloopSamPredictor(sam2_model)
        self.cache_items_dict = dict()

    @staticmethod
    def progress_update(progress, message):
        if progress is not None:
            progress.update(message=message)

    def cache_item(self, item: dl.Item):
        if item.id not in self.cache_items_dict:
            logger.info(f'item: {item.id} isnt cached, preparing...')
            image = item.download(save_locally=False, to_array=True, overwrite=True)
            setting_image_params = self.predictor.set_image(image=image)
            self.cache_items_dict[item.id] = CachedItem(image_embed=setting_image_params['image_embed'],
                                                        orig_hw=setting_image_params['orig_hw'],
                                                        high_res_feats=setting_image_params['high_res_feats'],
                                                        timestamp=datetime.datetime.now())

    def sam_predict_box(self,
                        item: dl.Item,
                        annotations,
                        return_type: str = 'segment',
                        progress: dl.Progress = None) -> list:
        """

        :param item:
        :param annotations:
        :param return_type:
        :param progress:
        :return:
        """
        logger.info(f'GPU available: {torch.cuda.is_available()}')
        tic_total = time.time()
        self.progress_update(progress=progress, message='Downloading item')
        logger.info('downloading item')

        self.progress_update(progress=progress, message='Running model')
        logger.info('running model')

        count = 1
        annotation_response = list()
        self.cache_item(item=item)
        image_params = self.cache_items_dict[item.id]
        for annotation_dict in annotations:
            annotation = dl.Annotation.from_json(annotation_dict)
            coordinates = annotation.coordinates
            logger.info(f'annotation {count}/{len(annotations)}')
            count += 1
            left = int(coordinates[0]['x'])
            top = int(coordinates[0]['y'])
            right = int(coordinates[1]['x'])
            bottom = int(coordinates[1]['y'])
            input_box = np.array([left, top, right, bottom])
            # Call SAM
            tic_model = time.time()
            masks, _, _ = self.predictor.predict(image_properties=image_params.dict(),
                                                 box=input_box,
                                                 multimask_output=False)
            toc_model = time.time()
            logger.info(f'time to get predicted mask: {round(toc_model - tic_model, 2)} seconds')

            #######################################
            mask = masks[0]
            model_info = {
                'name': 'sam2',
                'confidence': 1.0
            }

            ##########
            # Upload #
            ##########

            if return_type in ['binary', 'Semantic']:
                annotation_definition = dl.Segmentation(
                    geo=mask,
                    label=annotation.label,
                    attributes=annotation.attributes
                )
            elif return_type in ['segment', 'Polygon']:
                annotation_definition = dl.Polygon.from_segmentation(
                    mask=mask,
                    label=annotation.label,
                    attributes=annotation.attributes
                )
            else:
                raise ValueError('Unknown return type: {}'.format(return_type))
            builder = item.annotations.builder()
            builder.add(annotation_definition=annotation_definition,
                        automated=True,
                        model_info=model_info,
                        metadata=annotation.metadata)
            new_annotation = builder.annotations[0].to_json()
            new_annotation['id'] = annotation.id
            annotation_response.append(new_annotation)
        logger.info('updating progress')
        self.progress_update(progress=progress, message='Done!')
        logger.info('done')

        runtime_total = time.time() - tic_total
        logger.info('Runtime:')
        logger.info(f'Total: {runtime_total:02.1f}s')
        return annotation_response
