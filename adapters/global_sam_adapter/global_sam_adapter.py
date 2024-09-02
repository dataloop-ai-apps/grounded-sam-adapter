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

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from adapters.global_sam_adapter.sam2_handler import DataloopSamPredictor

logger = logging.getLogger('[GLOBAL-SAM]')
logger.setLevel('INFO')


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(
            self,
            cap,
            num_frames,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
    ):
        self.cap = cap
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self.images` will be loaded asynchronously
        self.images = [None] * num_frames
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        # wait for the video capture to be opened
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm.tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = threading.Thread(target=_load_frames, daemon=True)
        self.thread.start()
        self.thread.join()

    @staticmethod
    def _load_img_as_tensor(img_pil, image_size):
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size  # the original video size
        return img, video_height, video_width

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError(f"Failure in frame loading thread {self.exception}") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        ret, frame = self.cap.read()

        img, video_height, video_width = self._load_img_as_tensor(img_pil=Image.fromarray(frame),
                                                                  image_size=self.image_size)
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


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
        logger.info(f'GPU available: {torch.cuda.is_available()}')

        # model_cfg = "sam2_hiera_l.yaml"
        # model_cfg = "sam2_hiera_b+.yaml"
        model_cfg = "sam2_hiera_s.yaml"
        # weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_large.pt'
        # weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_base_plus.pt'
        weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_small.pt'
        # weights_filepath = 'artifacts/sam2_hiera_large.pt'
        # weights_filepath = 'artifacts/sam2_hiera_base_plus.pt'
        weights_filepath = 'artifacts/sam2_hiera_small.pt'
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
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        free = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.total --format=csv"
        info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        total = [int(x.split()[0]) for i, x in enumerate(info)]
        command = "nvidia-smi --query-gpu=memory.used --format=csv"
        info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        used = [int(x.split()[0]) for i, x in enumerate(info)]
        return free, total, used

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

    def _track_get_item_stream_capture(self, dl, item_stream_url):
        #############
        orig_item = None
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
        return cv2.VideoCapture('{}?jwt={}'.format(item_stream_url, dl.token())), orig_item

    @staticmethod
    def _track_calc_new_size(height, width, max_size):
        ratio = np.maximum(height, width) / max_size

        width, height = int(width / ratio), int(height / ratio)
        return width, height

    # Semantic studio function
    def get_sam_features(self, dl, item):
        self.cache_item(item=item)

        embedding = self.cache_items_dict[item.id].image_embed
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

    # default studio

    @staticmethod
    @torch.inference_mode()
    def init_state(
            self,
            cap,
            image_size,
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
            image_size=image_size,
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

    @staticmethod
    def _get_max_frame_duration(item, frame_duration, start_frame):
        if start_frame + frame_duration > int(item.metadata['system']['ffmpeg']['nb_read_frames']):
            frame_duration = int(item.metadata['system']['ffmpeg']['nb_read_frames']) - start_frame
        return frame_duration

    def track(self, dl, item_stream_url, bbs, start_frame, frame_duration=60, progress=None) -> dict:

        free, total, used = self.get_gpu_memory()
        logger.info(f'GPU memory - total: {total}, used: {used}, free: {free}')

        cap, orig_item = self._track_get_item_stream_capture(dl=dl, item_stream_url=item_stream_url)
        frame_duration = self._get_max_frame_duration(item=orig_item,
                                                      frame_duration=frame_duration,
                                                      start_frame=start_frame)
        logger.info(f"Setting cap to start frame {start_frame}")
        start_time = time.time()
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        end_time = start_time - time.time()
        max_retries = 3
        while end_time > 30 and max_retries > 0:
            logger.info(f"Retrying to set cap to start frame {start_frame}, retries left: {max_retries}")
            cap, _ = self._track_get_item_stream_capture(dl=dl, item_stream_url=item_stream_url)
            start_time = time.time()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            end_time = start_time - time.time()
            max_retries -= 1
        if end_time > 30:
            raise RuntimeError(f'Failed to get video stream {start_frame}')
        logger.info("Setting cap to start frame - Done")
        video_segments = {bbox_id: dict() for bbox_id, _ in bbs.items()}
        image_size = 1024  # must be same height and width
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.init_state(self=self.video_predictor,
                                              cap=cap,
                                              image_size=image_size,
                                              num_frames=frame_duration,
                                              offload_state_to_cpu=False,
                                              offload_video_to_cpu=False)

            for bbox_id, bb in bbs.items():
                left = int(bb[0]['x'])
                top = int(bb[0]['y'])
                right = int(bb[1]['x'])
                bottom = int(bb[1]['y'])
                input_box = np.array([left, top, right, bottom])
                frame_idx, object_ids, masks = self.video_predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=bbox_id,
                    box=input_box)
            # propagate the prompts to get masklets throughout the video
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                    inference_state=inference_state,
                    # start_frame_idx=start_frame,
                    # max_frame_num_to_track=frame_duration
            ):
                for i, bbox_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()[0]
                    if mask.any():
                        logger.info('Found tracking BB')
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)
                        ymin, ymax = np.where(rows)[0][[0, -1]]
                        xmin, xmax = np.where(cols)[0][[0, -1]]
                        video_segments[bbox_id][start_frame + out_frame_idx] = dl.Box(top=int(np.round(ymin)),
                                                                                      left=int(np.round(xmin)),
                                                                                      bottom=int(np.round(ymax)),
                                                                                      right=int(np.round(xmax)),
                                                                                      label='dummy').to_coordinates(
                            color=None)
                    else:
                        logger.info('NOT Found tracking BB')
                        video_segments[bbox_id][start_frame + out_frame_idx] = None
        return video_segments

    def box_to_segmentation(self,
                            dl,
                            item: dl.Item,
                            annotations,
                            progress: dl.Progress = None) -> list:
        return self.sam_predict_box(dl=dl, item=item, annotations=annotations, return_type='Semantic',
                                    progress=progress)

    def box_to_polygon(self,
                       dl,
                       item: dl.Item,
                       annotations,
                       progress: dl.Progress = None) -> list:
        return self.sam_predict_box(dl=dl, item=item, annotations=annotations, return_type='Polygon', progress=progress)

    def sam_predict_box(self,
                        dl,
                        item: dl.Item,
                        annotations,
                        return_type: str = 'segment',
                        progress: dl.Progress = None) -> list:
        """

        :param dl: DTLPY sdk instance
        :param item:
        :param annotations:
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
        if bb is not None:
            # The model can also take a box as input, provided in xyxy format.
            left = int(np.maximum(bb[0]['x'], 0))
            top = int(np.maximum(bb[0]['y'], 0))
            right = int(np.minimum(bb[1]['x'], image_params.orig_hw[1]))
            bottom = int(np.minimum(bb[1]['y'], image_params.orig_hw[0]))
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
        builder: dl.AnnotationCollection = item.annotations.builder()
        # boxed_mask = masks[0][bb[0]['y']:bb[1]['y'], bb[0]['x']:bb[1]['x']]
        boxed_mask = masks[0][input_box[1]:input_box[3], input_box[0]:input_box[2]]
        builder.add(annotation_definition=dl.Segmentation(geo=boxed_mask > 0, label='dummy'))
        toc_final = time.time()
        logger.info(f'time to create annotations: {round(toc_final - tic_3, 2)} seconds')
        logger.info(f'Total time of execution: {round(toc_final - tic_1, 2)} seconds')
        results = builder.annotations[0].annotation_definition.to_coordinates(color=color)
        return results
