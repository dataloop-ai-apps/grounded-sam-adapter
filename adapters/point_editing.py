import datetime
import torch
from segment_anything import sam_model_registry
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
        weights_filepath = 'artifacts1/sam_vit_h_4b8939.pth'
        model_type = "vit_h"
        device = "cuda"
        self.show = False
        if not os.path.isfile(weights_filepath):
            os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_filepath)
        sam = sam_model_registry[model_type](checkpoint=weights_filepath)
        sam.to(device=device)
        self.predictor = DataloopSamPredictor(sam)
        self.cache_items_dict = dict()

    def predict_interactive_editing(self, dl, item, points, bb=None, mask_uri=None, click_radius=4, color=None):
        """
        :param item: item to run on
        :param bb: ROI to crop bb[0]['y']: bb[1]['y'], bb[0]['x']: bb[1]['x']
        :param click_radius: ROI to crop
        :param mask_uri: ROI to crop
        :param points: list of {'x':x, 'y':y, 'in':True}
        :return:
        """
        # get item's image
        if 'bot.dataloop.ai' in dl.info()['user_email']:
            raise ValueError('This function cannot run with a bot user')
        tic_1 = time.time()
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
                                                        timestamp=datetime.datetime.now()
                                                        )
        image_params = self.cache_items_dict[item.id]
        toc_1 = time.time()
        logger.info(f'time to prepare  item: {round(toc_1 - tic_1, 2)} seconds')

        logger.info(f'Running prediction...')
        # get prediction
        tic_2 = time.time()
        results = None
        if bb is not None:
            # The model can also take a box as input, provided in xyxy format.
            input_box = np.array([bb[0]['x'], bb[0]['y'], bb[1]['x'], bb[1]['y']])
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
        boxed_mask = masks[0][bb[0]['y']:bb[1]['y'], bb[0]['x']:bb[1]['x']]
        builder.add(annotation_definition=dl.Segmentation(geo=boxed_mask > 0, label='dummy'))
        toc_final = time.time()
        logger.info(f'time to create annotations: {round(toc_final - tic_3, 2)} seconds')
        logger.info(f'Total time of execution: {round(toc_final - tic_1, 2)} seconds')
        results = builder.annotations[0].annotation_definition.to_coordinates(color=color)
        return results


def test():
    runner = Runner(dl=dl)
    bb = [{"x": 66,
           "y": 79},
          {"x": 287,
           "y": 460}
          ]
    points = [{"x": 177, "y": 270, "in": True}]
    item = dl.items.get(item_id='645378429033ec3fcd3b2036')
    color = [255, 0, 0]

    mask_coords = runner.predict_interactive_editing(dl, bb=bb, item=item, color=color, points=points)


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


def deploy():
    package_name = 'sam-point-editing'
    project_name = 'DataloopTasks'

    project = dl.projects.get(project_name=project_name)

    ##################
    # upload_artifacts
    ##################
    def upload_artifacts():
        project.artifacts.upload(filepath='weights/sam_vit_h_4b8939.pth',
                                 package_name=package_name)

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
                                                              name='predict_interactive_editing')
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
    service = package.services.deploy(service_name=package_name,
                                      init_input=[],
                                      module_name='sam',
                                      # sdk_version='1.76.7',
                                      runtime=dl.KubernetesRuntime(pod_type=dl.INSTANCE_CATALOG_GPU_K80_M,
                                                                   concurrency=5,
                                                                   autoscaler=dl.KubernetesRabbitmqAutoscaler(
                                                                       min_replicas=1),
                                                                   runner_image='gcr.io/viewo-g/piper/agent/runner/gpu/sam_point_edit:0.5.0'),
                                      is_global=True,
                                      jwt_forward=True
                                      )
    service = dl.services.get(service_name=package_name)
    service.package_revision = package.version
    service.update(force=True)


if __name__ == "__main__":
    dl.setenv('prod')
    # test()
    # deploy()
