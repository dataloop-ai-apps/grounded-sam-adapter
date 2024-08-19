import urllib.request
import collections
import subprocess
import pydantic
import datetime
import logging
import base64
import torch
import json
import time
import os
import cv2

import numpy as np
import dtlpy as dl

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from adapters.global_sam_adapter.sam2_handler import DataloopSamPredictor
from adapters.global_sam_adapter.tracked_box import TrackedBox, Bbox, AsyncVideoFrameLoader

logger = logging.getLogger('[GLOBAL-SAM]')
logger.setLevel('INFO')


class CachedItem(pydantic.BaseModel):
    image_embed: torch.Tensor
    timestamp: datetime.datetime
    high_res_feats: tuple
    orig_hw: tuple

    class Config:
        arbitrary_types_allowed = True


class Runner(dl.BaseServiceRunner):
    def __init__(self, dl):
        """
        Init package attributes here

        :return:
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model_cfg = os.path.join(os.getcwd(), "utils", "sam2_hiera_l.yaml")
        weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_large.pt'
        weights_filepath = 'artifacts/sam2_hiera_large.pth'
        self.show = False
        if not os.path.isfile(weights_filepath):
            os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_filepath)

        sam2_model = build_sam2(model_cfg, weights_filepath, device=device)
        self.predictor = DataloopSamPredictor(sam2_model)
        self.video_predictor: SAM2VideoPredictor = build_sam2_video_predictor(model_cfg, weights_filepath,
                                                                              device=device)
        self.cache_items_dict = dict()
        # tracker params
        self.MAX_AGE = 20
        self.THRESH = 0.4
        self.MIN_AREA = 20

    @staticmethod
    def get_gpu_memory():
        output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
        COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
        try:
            memory_use_info = output_to_list(subprocess.check_output(COMMAND.split(),
                                                                     stderr=subprocess.STDOUT))[1:]
        except subprocess.CalledProcessError as e:
            raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
        # print(memory_use_values)
        return memory_use_values

    @staticmethod
    def progress_update(progress, message):
        if progress is not None:
            progress.update(message=message)

    def get_sam_features(self, dl, item, to_upload):
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

    def cache_item(self, item: dl.Item):
        if item.id not in self.cache_items_dict:
            logger.info(f'item: {item.id} isnt cached, preparing...')
            image = item.download(save_locally=False, to_array=True, overwrite=True)
            setting_image_params = self.predictor.set_image(image=image)
            self.cache_items_dict[item.id] = CachedItem(image_embed=setting_image_params['image_embed'],
                                                        orig_hw=setting_image_params['orig_hw'],
                                                        high_res_feats=setting_image_params['high_res_feats'],
                                                        timestamp=datetime.datetime.now())

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

    @staticmethod
    def _track_get_modality(mod: dict):
        if 'operation' in mod:
            if mod['operation'] == 'replace':
                return mod['itemId']
        elif 'type' in mod:
            if mod['type'] == 'replace':
                return mod['ref']
        else:
            return None

    def _track_get_item_stream_capture(self, item_stream_url):
        #############
        # replace to webm stream
        if dl.environment() in item_stream_url:
            # is dataloop stream - take webm
            item_id = item_stream_url[item_stream_url.find('items/') + len('items/'): -7]
            orig_item = dl.items.get(item_id=item_id)
            webm_id = None
            for mod in orig_item.metadata['system'].get('modalities', list()):
                ref = self._track_get_modality(mod)
                if ref is not None:
                    try:
                        _ = dl.items.get(item_id=ref)
                        webm_id = ref
                        break
                    except dl.exceptions.NotFound:
                        continue
            if webm_id is not None:
                # take webm if exists
                item_stream_url = item_stream_url.replace(item_id, webm_id)
        ############
        return cv2.VideoCapture('{}?jwt={}'.format(item_stream_url, dl.token()))

    @staticmethod
    @torch.inference_mode()
    def init_state(
            self,
            cap,
            num_frames,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        images = AsyncVideoFrameLoader(
            cap=cap,
            num_frames=num_frames,
            img_mean=img_mean,
            img_std=img_std,
            image_size=1024,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=compute_device,
        )

        video_height, video_width = images.video_height, images.video_width

        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = collections.OrderedDict()
        inference_state["obj_idx_to_id"] = collections.OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def track_new(self, dl, item_stream_url, bbs, start_frame, frame_duration=60, progress=None) -> dict:
        import torch
        video_segments = dict()

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            cap = self._track_get_item_stream_capture(item_stream_url)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            inference_state = self.init_state(self=self.video_predictor,
                                              cap=cap,
                                              num_frames=frame_duration,
                                              )

            for bbox_id, bb in bbs.items():
                left = int(bb[0]['x'])
                top = int(bb[0]['y'])
                right = int(bb[1]['x'])
                bottom = int(bb[1]['y'])
                input_box = np.array([left, top, right, bottom])
                frame_idx, object_ids, masks = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=start_frame,
                    obj_id=bbox_id,
                    box=input_box)
            # propagate the prompts to get masklets throughout the video
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
        return video_segments

    def track(self, dl, item_stream_url, bbs, start_frame, frame_duration=60, progress=None) -> dict:
        """
        :param item_stream_url:  item.stream for Dataloop item, url for json video links
        :param bbs: dictionary of annotation.id : BB
        :param start_frame:
        :param frame_duration:
        :param progress:
        :return:
        """
        try:
            logger.info(f'GPU memory usage: {self.get_gpu_memory()}[mb]')

            if not isinstance(bbs, dict):
                raise ValueError('input "bbs" must be a dictionary of {id:bbox}')
            logger.info('[Tracker] Started')

            logger.info('[Tracker] video url: {}'.format(item_stream_url))
            d_size = 1024
            tic_get_cap = time.time()
            cap = self._track_get_item_stream_capture(item_stream_url)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            x_factor = frame_width / d_size
            y_factor = frame_height / d_size
            runtime_get_cap = time.time() - tic_get_cap
            logger.info('[Tracker] starting from {} to {}'.format(start_frame, start_frame + frame_duration))

            logger.info('[Tracker] received bbs(xyxy): {}'.format(bbs))
            runtime_load_frame = list()
            runtime_track = list()

            tic_total = time.time()
            output_dict = {bbox_id: dict() for bbox_id, _ in bbs.items()}
            states_dict = {bbox_id: TrackedBox(Bbox.from_xyxy(bb[0]['x'] / x_factor,
                                                              bb[0]['y'] / y_factor,
                                                              bb[1]['x'] / x_factor,
                                                              bb[1]['y'] / y_factor),
                                               max_age=self.MAX_AGE) for bbox_id, bb in bbs.items()}

            logger.info('[Tracker] going to process {} frames'.format(frame_duration))
            for i_frame in range(1, frame_duration):
                logger.info(f'GPU memory usage: {self.get_gpu_memory()}[mb]')

                logger.info('[Tracker] processing frame #{}'.format(start_frame + i_frame))
                tic = time.time()
                ret, frame = cap.read()
                states_dict_flag = all(bb.gone for bb in states_dict.values())
                if not ret or states_dict_flag:
                    logger.info(f"[Tracker] stopped at frame {i_frame}: "
                                f"opencv frame read :{ret}, all bbs gone: {states_dict_flag}")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_params: dict = self.predictor.set_image(image=cv2.resize(frame, (d_size, d_size)))

                runtime_load_frame.append(time.time() - tic)

                tic = time.time()
                for bbox_id, bb in bbs.items():
                    logger.info(f'GPU memory usage: {self.get_gpu_memory()}[mb]')

                    # track
                    states_dict[bbox_id].track(sam=self.predictor,
                                               image_params=image_params,
                                               thresh=self.THRESH,
                                               min_area=self.MIN_AREA)
                    bbox = states_dict[bbox_id].bbox
                    if bbox is None:
                        logger.info('NOT Found tracking BB')
                        output_dict[bbox_id][start_frame + i_frame] = None
                    else:
                        logger.info('Found tracking BB')
                        output_dict[bbox_id][start_frame + i_frame] = dl.Box(top=int(np.round(bbox.y * y_factor)),
                                                                             left=int(np.round(bbox.x * x_factor)),
                                                                             bottom=int(np.round(bbox.y2 * y_factor)),
                                                                             right=int(np.round(bbox.x2 * x_factor)),
                                                                             label='dummy').to_coordinates(color=None)

                runtime_track.append(time.time() - tic)

            runtime_total = time.time() - tic_total
            fps = frame_duration / (runtime_total + 1e-6)
            logger.info('[Tracker] Finished.')
            logger.info('[Tracker] Runtime information: \n'
                        f'Total runtime: {runtime_total:.2f}[s]\n'
                        f'FPS: {fps:.2f}fps\n'
                        f'Get url capture object: {runtime_get_cap:.2f}[s]\n'
                        f'Total track time: {np.sum(runtime_load_frame) + np.sum(runtime_track):.2f}[s]\n'
                        f'Mean load per frame: {np.mean(runtime_load_frame):.2f}\n'
                        f'Mean track per frame: {np.mean(runtime_track):.2f}')
        except Exception:
            logger.exception('Failed during track:')
            raise
        return output_dict

    def box_to_seg(self,
                   dl,
                   item: dl.Item,
                   annotations_dict: dict,
                   return_type: str = 'segment',
                   progress: dl.Progress = None) -> list:
        """

        :param dl: DTLPY sdk instance
        :param item:
        :param annotations_dict:
        :param return_type:
        :param progress:
        :return:
        """
        # get item's image
        if 'bot.dataloop.ai' in dl.info()['user_email']:
            raise ValueError('This function cannot run with a bot user')

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
        for annotation_id, annotation in annotations_dict.items():
            coordinates = annotation.coordinates
            logger.info(f'annotation {count}/{len(annotations_dict)}')
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
                'name': 'siammask',
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


def test():
    # ex = dl.executions.get('65321a98e808fdceac4a6fe6')
    self = Runner(dl=dl)
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
    self.cache_item(item=item)
    embedding = self.cache_items_dict[item.id].image_embed
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


def test_box_to_seg():
    self = Runner(dl=dl)
    item = dl.items.get(None, '659c029a9d62ab2514292e87')
    annotations_dict = {ann.id: ann for ann in item.annotations.list() if ann.type == 'box'}
    # mask_coords = runner.predict_interactive_editing(dl, item=item, **ex.input)
    # emb = runner.get_sam_features(dl=dl, item=item)
    annotations = self.box_to_seg(dl=dl, item=item, annotations_dict=annotations_dict, return_type='binary')
    item.annotations.upload(annotations)


def test_tracker():
    self = Runner(dl=dl)
    item = dl.items.get(item_id='66c32175b0aedce631ccd4b1')
    inputs = {
        "item_stream_url": item.stream,
        "bbs": {ann.id: ann.coordinates for ann in item.annotations.list() if ann.type == 'box'},
        "start_frame": 0,
        "frame_duration": 72,
        "dl": dl
    }

    # inputs = dl.executions.get('66c327b8945b576b92ea2754').input
    # inputs['dl'] = dl
    # output_dict = self.track(**inputs)
    output_dict = self.track_new(**inputs)
    for a_id, frames in output_dict.items():
        annotation = dl.annotations.get(a_id)
        for i_frame, box in frames.items():
            annotation.add_frame(frame_num=i_frame,
                                 annotation_definition=dl.Box(left=box[0]['x'],
                                                              right=box[1]['x'],
                                                              top=box[0]['y'],
                                                              bottom=box[1]['y'],
                                                              label=annotation.label))
        annotation.update(True)


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
    # test()
    # deploy()
    # runner = Runner(dl=dl)
    # test_box_to_seg()
    test_tracker()
