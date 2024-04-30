import datetime
import json
import base64
import struct
import torch
from segment_anything import sam_model_registry, SamPredictor
from adapters.sam_handler import DataloopSamPredictor
import dtlpy as dl
import numpy as np
import os
import time
import logging
import urllib.request

from pydantic import BaseModel

logger = logging.getLogger()


class CachedItem(BaseModel):
    image_embeddings: torch.Tensor
    timestamp: datetime.datetime
    original_size: tuple
    input_size: tuple

    class Config:
        arbitrary_types_allowed = True


class Runner(dl.BaseServiceRunner):
    def __init__(self, dl):
        """
        Init package attributes here

        :return:
        """
        weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam/sam_vit_h_4b8939.pth'
        # weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam/sam_vit_b_01ec64.pth'
        weights_filepath = 'artifacts/sam_vit_h_4b8939.pth'
        # weights_filepath = 'artifacts/sam_vit_b_01ec64.pth'
        model_type = "vit_h"
        # model_type = "vit_b"
        device = "cpu"
        self.show = False
        if not os.path.isfile(weights_filepath):
            os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_filepath)
        sam = sam_model_registry[model_type](checkpoint=weights_filepath)
        sam.to(device=device)
        self.predictor = DataloopSamPredictor(sam)
        self.cache_items_dict = dict()

    def get_sam_features(self, dl, item):
        self.cache_item(item=item)
        embedding = self.cache_items_dict[item.id].image_embeddings
        bytearray_data = embedding.cpu().numpy().tobytes()  # float32
        base64_str = base64.b64encode(bytearray_data).decode('utf-8')
        if not os.path.isdir('tmp'):
            os.makedirs('tmp')
        with open(f'tmp/{item.id}.json', 'w') as f:
            json.dump({'item': base64_str}, f)
        features_item = item.dataset.items.upload(local_path=f'tmp/{item.id}.json',
                                                  remote_path='/.dataloop/sam_features',
                                                  overwrite=True,
                                                  remote_name=f'{item.id}.json')
        return features_item.id

    @staticmethod
    def get_or_create_feature_set(item: dl.Item):
        project = item.project
        try:
            feature_set = project.feature_sets.get(feature_set_name='sam_vit_h_image_embeddings')
        except dl.exceptions.NotFound:
            feature_set = project.feature_sets.create(name='sam_vit_h_image_embeddings',
                                                      size=1 * 256 * 64 * 64,
                                                      set_type='sam',
                                                      entity_type=dl.FeatureEntityType.ITEM)
        return feature_set

    def cache_item(self, item: dl.Item):
        if item.id not in self.cache_items_dict:
            logger.info(f'item: {item.id} isnt cached, preparing...')
            image = item.download(save_locally=False, to_array=True, overwrite=True)
            setting_image_params = self.predictor.set_image(image=image)
            # {'image_embeddings': self.model.image_encoder(input_image),
            #  'original_size': original_image_size,
            #  'input_size': tuple(transformed_image.shape[-2:])}
            self.cache_items_dict[item.id] = CachedItem(image_embeddings=setting_image_params['image_embeddings'],
                                                        original_size=setting_image_params['original_size'],
                                                        input_size=setting_image_params['input_size'],
                                                        timestamp=datetime.datetime.now())
            ####### NOT WORKING
            # feature_set = self.get_or_create_feature_set(item=item)
            # filters = dl.Filters(resource=dl.FiltersResource.FEATURE)
            # filters.add(field='featureSetId', values=feature_set.id)
            # filters.add(field='entityId', values=item.id)
            # pages = item.features.list(filters=filters)
            # if pages.items_count == 0:
            #     logger.info(f'item: {item.id} isnt cached, preparing...')
            #     image = item.download(save_locally=False, to_array=True, overwrite=True)
            #     setting_image_params = self.predictor.set_image(image=image)
            #     # {'image_embeddings': self.model.image_encoder(input_image),
            #     #  'original_size': original_image_size,
            #     #  'input_size': tuple(transformed_image.shape[-2:])}
            #     embeddings = setting_image_params['image_embeddings']
            #     feature = feature_set.features.create(value=embeddings.numpy().flatten().tolist(),
            #                                           project_id=item.project_id,
            #                                           entity_id=item.id)
            #     self.cache_items_dict[item.id] = CachedItem(image_embeddings=setting_image_params['image_embeddings'],
            #                                                 original_size=setting_image_params['original_size'],
            #                                                 input_size=setting_image_params['input_size'],
            #                                                 timestamp=datetime.datetime.now(),
            #                                                 feature_vector_id=feature.id)
            # else:
            #     feature_vector = pages.items[0]
            #     self.cache_items_dict[item.id] = CachedItem(image_embeddings=feature_vector.value,
            #                                                 original_size=setting_image_params['original_size'],
            #                                                 input_size=setting_image_params['input_size'],
            #                                                 timestamp=datetime.datetime.now(),
            #                                                 feature_vector_id=feature_vector.id)

    def predict_interactive_editing(self, dl, item, points, bb=None, mask_uri=None, click_radius=4, color=None):
        """
        :param item: item to run on
        :param bb: ROI to crop bb[0]['y']: bb[1]['y'], bb[0]['x']: bb[1]['x']
        :param click_radius: ROI to crop
        :param mask_uri: ROI to crop
        :param points: list of {'x':x, 'y':y, 'in':True}
        :param color:
        :return:
        """
        # get item's image
        if 'bot.dataloop.ai' in dl.info()['user_email']:
            raise ValueError('This function cannot run with a bot user')
        tic_1 = time.time()
        self.cache_item(item=item)

        image_params = self.cache_items_dict[item.id]
        toc_1 = time.time()
        logger.info(f'time to prepare  item: {round(toc_1 - tic_1, 2)} seconds')

        logger.info(f'Running prediction...')
        # get prediction
        tic_2 = time.time()
        results = None
        if bb is not None:
            # The model can also take a box as input, provided in xyxy format.
            left = int(np.maximum(bb[0]['x'], 0))
            top = int(np.maximum(bb[0]['y'], 0))
            right = int(np.minimum(bb[1]['x'], image_params.original_size[1]))
            bottom = int(np.minimum(bb[1]['y'], image_params.original_size[0]))
            # input_box = np.array([bb[0]['x'], bb[0]['y'], bb[1]['x'], bb[1]['y']])
            input_box = np.array([left, top, right, bottom])
        else:
            input_box = None

        if len(points) > 0:
            point_coords = list()
            point_labels = list()
            for pt in points:
                point_labels.append(1 if pt['in'] is True else 0)
                point_coords.append([pt['x'], pt['y']])
            point_labels = np.asarray(point_labels)
            point_coords = np.asarray(point_coords)
        else:
            point_labels = None
            point_coords = None

        masks, _, _ = self.predictor.predict(image_properties=image_params.dict(),
                                             point_coords=point_coords,
                                             point_labels=point_labels,
                                             box=input_box,
                                             multimask_output=False,
                                             )
        toc_2 = time.time()
        logger.info(f'time to get predicted mask: {round(toc_2 - tic_2, 2)} seconds')

        # push new annotation
        logger.info(f'Creating new predicted mask...')
        tic_3 = time.time()
        builder = item.annotations.builder()  # type: dl.AnnotationCollection
        # boxed_mask = masks[0][bb[0]['y']:bb[1]['y'], bb[0]['x']:bb[1]['x']]
        boxed_mask = masks[0][input_box[1]:input_box[3], input_box[0]:input_box[2]]
        builder.add(annotation_definition=dl.Segmentation(geo=boxed_mask > 0, label='dummy'))
        toc_final = time.time()
        logger.info(f'time to create annotations: {round(toc_final - tic_3, 2)} seconds')
        logger.info(f'Total time of execution: {round(toc_final - tic_1, 2)} seconds')
        results = builder.annotations[0].annotation_definition.to_coordinates(color=color)
        return results


def test():
    # ex = dl.executions.get('65321a98e808fdceac4a6fe6')
    runner = Runner(dl=dl)
    # bb = [{"x": 66,
    #        "y": 79},
    #       {"x": 287,
    #        "y": 460}
    #       ]
    # points = [{"x": 177, "y": 270, "in": True}]
    # color = [255, 0, 0]
    #
    # item = dl.items.get(item_id=ex.input.pop('item')['item_id'])
    # item = dl.items.get(None, '652d050fd73711801c5d6120')
    item = dl.items.get(None, '659c0f5ae86a6a3c2e97d7c8')
    # mask_coords = runner.predict_interactive_editing(dl, item=item, **ex.input)
    # emb = runner.get_sam_features(dl=dl, item=item)
    runner.cache_item(item=item)
    embedding = runner.cache_items_dict[item.id].image_embeddings
    bytearray_data = bytearray(embedding.cpu().numpy().tobytes())
    # float_list = struct.unpack('f' * (len(bytearray_data) // 4), bytearray_data)
    # binary_data = struct.pack('f' * len(float_list), *float_list)
    base64_str = base64.b64encode(bytearray_data).decode('utf-8')
    with open(r'e:\ttt.json', 'w') as f:
        json.dump(base64_str, f)


def test_ex():
    service = dl.services.get(service_name='sam-point-editing')
    ex = service.execute(
        function_name='predict_interactive_editing',
        execution_input={"bb": [{"x": 65, "y": 91},
                                {"x": 282, "y": 463}],
                         "points": [{"x": 174, "y": 277, "in": True}],
                         "item": {"item_id": "64e5f716fe649e509dc98351"},
                         "color": [255, 0, 0]}
    )
    service = dl.services.get(service_name='sam-point-editing')
    ex = service.execute(
        function_name='get_sam_features',
        execution_input={"item": {"item_id": "652d050fd73711801c5d6120"}}
    )


def deploy():
    package_name = 'sam-point-editing'
    project_name = 'DataloopTasks'

    project = dl.projects.get(project_name=project_name)


    ##################
    # push package
    ##################
    modules = [dl.PackageModule(entry_point='adapters/point_editing.py',
                                class_name='Runner',
                                name='sam',
                                init_inputs=[],
                                functions=[dl.PackageFunction(inputs=[dl.FunctionIO(type='Item', name='item'),
                                                                      dl.FunctionIO(type='Json', name='bb'),
                                                                      dl.FunctionIO(type='Json', name='points'),
                                                                      dl.FunctionIO(type='Json', name='color')],
                                                              name='predict_interactive_editing'),
                                           dl.PackageFunction(inputs=[dl.FunctionIO(type='Item', name='item')],
                                                              name='get_sam_features')
                                           ]
                                )
               ]
    package = project.packages.push(package_name=package_name,
                                    src_path=os.getcwd(),
                                    modules=modules,
                                    requirements=[dl.PackageRequirement(name='dtlpy')],
                                    ignore_sanity_check=True)
    # package = project.packages.get(package_name=package_name)

    ##################
    # deploy service
    ##################
    # service = package.services.deploy(service_name=package_name,
    #                                   init_input=[],
    #                                   module_name='sam',
    #                                   # sdk_version='1.76.7',
    #                                   runtime=dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_M,
    #                                                                concurrency=5,
    #                                                                autoscaler=dl.KubernetesRabbitmqAutoscaler(
    #                                                                    min_replicas=1),
    #                                                                runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/sam_point_edit:0.5.0'),
    #                                   is_global=True,
    #                                   jwt_forward=True
    #                                   )
    service = dl.services.get(service_name=package_name)
    service.package_revision = package.version
    service.update(force=True)


if __name__ == "__main__":
    dl.setenv('rc')
    test()
    # deploy()
