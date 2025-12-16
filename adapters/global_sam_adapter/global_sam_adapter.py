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
import os
import cv2
import dtlpy as su_dl
import numpy as np

from sam3.model_builder import build_sam3_video_predictor
from sam3.model.sam3_video_predictor import Sam3VideoPredictor

from adapters.global_sam_adapter.sam3_handler import DataloopSamPredictor

logger = logging.getLogger('[GLOBAL-SAM]')
logger.setLevel('INFO')

# Enable cuDNN autotuner for fixed input sizes to improve convolution performance
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass


class AsyncVideoFrameLoader:
    """
    Asynchronously prefetches video frames without blocking session start.
    Uses a single producer thread to read from the capture in order, while
    consumers wait on availability via a condition variable.
    """

    def __init__(self, cap, num_frames, image_size, offload_video_to_cpu, img_mean, img_std, compute_device):
        self.cap = cap
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.compute_device = compute_device
        # keep mean/std on the same device as frames to avoid CPU math
        self.img_mean = img_mean.to(self.compute_device, non_blocking=True)
        self.img_std = img_std.to(self.compute_device, non_blocking=True)

        # items in `self.images` will be loaded asynchronously
        self.images = [None] * num_frames
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width filled when loading the first frame
        self.video_height = None
        self.video_width = None

        # condition variable for producer/consumer coordination
        self._cv = threading.Condition()
        self._next_to_fill = 0
        self._stop = False

        # synchronously read and process the first frame to set sizes
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read first frame from capture")
        self.video_height, self.video_width = frame.shape[0], frame.shape[1]
        self.images[0] = self._frame_to_tensor(frame)
        self._next_to_fill = 1

        # start background producer for remaining frames
        self.thread = threading.Thread(target=self._prefetch, daemon=True)
        self.thread.start()

    def _frame_to_tensor(self, frame_np: np.ndarray) -> torch.Tensor:
        # OpenCV provides BGR; convert to RGB (use getattr to avoid linter cv2 stub issues)
        cvtColor = getattr(cv2, 'cvtColor')
        COLOR_BGR2RGB = getattr(cv2, 'COLOR_BGR2RGB', 4)
        INTER_LINEAR = getattr(cv2, 'INTER_LINEAR', 1)
        resize = getattr(cv2, 'resize')
        rgb = cvtColor(frame_np, COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.image_size, self.image_size):
            rgb = resize(rgb, (self.image_size, self.image_size), interpolation=INTER_LINEAR)
        # HWC uint8 -> CHW float32 [0,1]
        img = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.compute_device, non_blocking=True)
        img = img.permute(2, 0, 1).float().div_(255.0)
        # normalize on device
        img.sub_(self.img_mean).div_(self.img_std)
        # optionally keep frames on CPU to save GPU memory
        if self.offload_video_to_cpu:
            img = img.to("cpu", non_blocking=True)
        return img

    def _prefetch(self):
        try:
            while True:
                with self._cv:
                    if self._stop or self._next_to_fill >= len(self.images):
                        self._cv.notify_all()
                        return
                    idx = self._next_to_fill
                    self._next_to_fill += 1
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise RuntimeError(f"Failed to read frame {idx}")
                img = self._frame_to_tensor(frame)
                with self._cv:
                    self.images[idx] = img
                    self._cv.notify_all()
        except Exception as e:
            with self._cv:
                self.exception = e
                self._cv.notify_all()

    def __getitem__(self, index):
        if index < 0 or index >= len(self.images):
            raise IndexError(index)
        with self._cv:
            while self.images[index] is None and self.exception is None:
                self._cv.wait(timeout=0.5)
            if self.exception is not None:
                raise RuntimeError(f"Failure in frame loading thread {self.exception}") from self.exception
            return self.images[index]

    def __len__(self):
        return len(self.images)


class CachedItem(pydantic.BaseModel):
    image_embed: torch.Tensor
    timestamp: datetime.datetime
    high_res_feats: tuple
    orig_hw: tuple

    class Config:
        arbitrary_types_allowed = True


class Runner(su_dl.BaseServiceRunner):
    def __init__(self, dl):
        """
        Init package attributes here

        :return:
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'GPU available: {torch.cuda.is_available()}')

        # SAM3 uses built-in model loading from HuggingFace by default
        # Build video predictor which properly loads all weights
        self.video_predictor: Sam3VideoPredictor = build_sam3_video_predictor()
        
        # Get the tracker and inject the detector's backbone into it
        # The tracker doesn't have its own backbone, but needs one for forward_image
        tracker_model = self.video_predictor.model.tracker
        tracker_model.backbone = self.video_predictor.model.detector.backbone
        self.predictor = DataloopSamPredictor(tracker_model)
        self.tracker = tracker_model
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

    def cache_item(self, item: su_dl.Item):
        if item.id not in self.cache_items_dict:
            logger.info(f'item: {item.id} isnt cached, preparing...')
            image = item.download(save_locally=False, to_array=True, overwrite=True)
            setting_image_params = self.predictor.set_image(image=image)
            self.cache_items_dict[item.id] = CachedItem(
                image_embed=setting_image_params['image_embed'],
                orig_hw=setting_image_params['orig_hw'],
                high_res_feats=setting_image_params['high_res_feats'],
                timestamp=datetime.datetime.now(),
            )

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
            item_id = item_stream_url[item_stream_url.find('items/') + len('items/') : -7]
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
        # use getattr to avoid static analyzer errors on cv2 stubs
        VideoCapture = getattr(cv2, "VideoCapture")
        return VideoCapture('{}?jwt={}'.format(item_stream_url, dl.token())), orig_item

    @staticmethod
    def _track_calc_new_size(height, width, max_size):
        ratio = np.maximum(height, width) / max_size

        width, height = int(width / ratio), int(height / ratio)
        return width, height

    # Semantic studio function
    def get_sam_features(self, dl, item):
        free, total, used = self.get_gpu_memory()
        print(f'Before inference GPU memory - total: {total}, used: {used}, free: {free}')

        self.cache_item(item=item)

        bytearray_data = self.cache_items_dict[item.id].image_embed.cpu().numpy().tobytes()  # float32
        base64_bytearray_data = base64.b64encode(bytearray_data).decode('utf-8')
        high_res_feats_0 = self.cache_items_dict[item.id].high_res_feats[0].cpu().numpy().tobytes()
        base64_bytearray_high_res_feats_0 = base64.b64encode(high_res_feats_0).decode('utf-8')
        high_res_feats_1 = self.cache_items_dict[item.id].high_res_feats[1].cpu().numpy().tobytes()
        base64_bytearray_high_res_feats_1 = base64.b64encode(high_res_feats_1).decode('utf-8')
        feeds = {
            'image_embed': base64_bytearray_data,
            'high_res_feats_0': base64_bytearray_high_res_feats_0,
            'high_res_feats_1': base64_bytearray_high_res_feats_1,
        }
        if not os.path.isdir('tmp'):
            os.makedirs('tmp')
        with open(f'tmp/{item.id}.json', 'w') as f:
            json.dump(feeds, f)

        su_ds = su_dl.datasets.get(dataset_id=item.dataset_id, fetch=False)
        features_item = su_ds.items.upload(
            local_path=f'tmp/{item.id}.json',
            remote_path='/.dataloop/sam_features',
            overwrite=True,
            remote_name=f'{item.id}.json',
        )
        return features_item.id

    # default studio

    @torch.inference_mode()
    def init_state(
        self, cap, image_size, num_frames, offload_video_to_cpu=False, offload_state_to_cpu=False
    ):
        """Initialize an inference state."""
        # SAM3 uses the tracker from the video model
        compute_device = self.tracker.device  # device of the model
        img_mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)[:, None, None]  # SAM3 uses different normalization
        img_std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float32)[:, None, None]
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

        # Use SAM3 tracker's init_state method
        inference_state = self.tracker.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu,
            async_loading_frames=False,
        )
        inference_state["images"] = images
        inference_state["device"] = compute_device
        
        # Warm up the visual backbone and cache the image feature on frame 0
        self.tracker._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    def track(self, dl, item_stream_url, bbs, start_frame, frame_duration=60, progress=None) -> dict:

        start_time_all = time.time()
        free, total, used = self.get_gpu_memory()
        logger.info(f'GPU memory - total: {total}, used: {used}, free: {free}')

        start_time = time.time()
        logger.info(f"Setting cap to start frame {start_frame}")

        video_loaded = False
        for _ in range(3):
            cap = None
            cap, orig_item = self._track_get_item_stream_capture(dl=dl, item_stream_url=item_stream_url)
            if cap is not None and cap.isOpened():
                CAP_PROP_POS_FRAMES = getattr(cv2, "CAP_PROP_POS_FRAMES")
                cap.set(CAP_PROP_POS_FRAMES, start_frame)
                if int(cap.get(CAP_PROP_POS_FRAMES)) == start_frame:
                    video_loaded = True
                    break
        if not video_loaded:
            raise RuntimeError(f'Failed to get video stream {item_stream_url} with start frame {start_frame}')

        try:
            n_frames = int(orig_item.metadata['system']['ffmpeg']['nb_read_frames'])
        except KeyError:
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame + frame_duration > n_frames:
            frame_duration = n_frames - start_frame

        logger.info("Setting cap to start frame - Done")
        logger.info(f"RUNTIME: set video cap: {time.time() - start_time:.2f}[s]")
        video_segments = {bbox_id: dict() for bbox_id, _ in bbs.items()}
        # match the model's preferred image size for better throughput/memory balance
        # SAM3 uses 1008 as default image size
        image_size = getattr(self.video_predictor.model, 'image_size', 1008)
        start_time = time.time()
        logger.info(f"Init start and loading frames...")
        use_cuda = torch.cuda.is_available()
        # prefer bf16 when supported, else fp16 on CUDA, else fp32 on CPU
        bf16_supported = getattr(torch.cuda, 'is_bf16_supported', lambda: False)() if use_cuda else False
        autocast_dtype = torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else torch.float32)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_cuda):
            inference_state = self.init_state(
                cap=cap,
                image_size=image_size,
                num_frames=frame_duration,
                offload_state_to_cpu=False,
                offload_video_to_cpu=False,
            )

            bbs_type_map = dict()
            for bbox_id, bb in bbs.items():
                if isinstance(bb, list) and isinstance(bb[0], list):
                    # flat the list in a list to just one list
                    bbs[bbox_id] = bb[0]
                    bbs_type_map[bbox_id] = 'polygon'
                elif isinstance(bb, list) and len(bb) > 2:
                    bbs_type_map[bbox_id] = 'polygon'
                else:
                    bbs_type_map[bbox_id] = 'box'

            for bbox_id, bb in bbs.items():
                left = int(np.min([pt['x'] for pt in bb]))
                top = int(np.min([pt['y'] for pt in bb]))
                right = int(np.max([pt['x'] for pt in bb]))
                bottom = int(np.max([pt['y'] for pt in bb]))
                input_box = np.array([left, top, right, bottom])
                self.tracker.add_new_points_or_box(
                    inference_state=inference_state, frame_idx=0, obj_id=bbox_id, box=input_box,
                    rel_coordinates=False, normalize_coords=False
                )
            logger.info(f"Init start and loading frames - Done")
            logger.info(f"RUNTIME: init state: {time.time() - start_time:.2f}[s]")
            start_time = time.time()
            logger.info(f"Propagate in video...")
            # propagate the prompts to get masklets throughout the video
            for out_frame_idx, out_obj_ids, _, out_mask_logits, _ in self.tracker.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=frame_duration,
                reverse=False,
                propagate_preflight=True,
            ):
                if out_frame_idx >= frame_duration:
                    break
                for i, bbox_id in enumerate(out_obj_ids):
                    # boolean mask on device; squeeze channel dim
                    mask_bool_2d = (out_mask_logits[i] > 0.0)[0]
                    if torch.any(mask_bool_2d):
                        # noisy: keep at debug to reduce overhead on long runs
                        logger.debug('Found tracking BB')
                        # fast bbox on device when output is box; otherwise convert for polygon
                        rows_any = torch.any(mask_bool_2d, dim=1)
                        cols_any = torch.any(mask_bool_2d, dim=0)
                        if bbs_type_map[bbox_id] == 'box':
                            y_indices = torch.where(rows_any)[0]
                            x_indices = torch.where(cols_any)[0]
                            ymin = int(y_indices[0].item())
                            ymax = int(y_indices[-1].item())
                            xmin = int(x_indices[0].item())
                            xmax = int(x_indices[-1].item())
                            video_segments[bbox_id][start_frame + out_frame_idx] = dl.Box(
                                top=int(np.round(ymin)),
                                left=int(np.round(xmin)),
                                bottom=int(np.round(ymax)),
                                right=int(np.round(xmax)),
                                label='dummy',
                            ).to_coordinates(color=None)
                        elif bbs_type_map[bbox_id] == 'polygon':
                            mask = mask_bool_2d.detach().cpu().numpy()
                            video_segments[bbox_id][start_frame + out_frame_idx] = dl.Polygon.from_segmentation(
                                mask=mask, label='dummy'
                            ).to_coordinates(color=None)[0]
                        else:
                            raise ValueError(f'Unknown annotation type: {bbs_type_map[bbox_id]}')
                    else:
                        logger.debug('NOT Found tracking BB')
                        # Don't remove
                        # Taking annotation from last frame if not found
                        video_segments[bbox_id][start_frame + out_frame_idx] = video_segments[bbox_id][
                            start_frame + out_frame_idx - 1
                        ]
            logger.info(f"Propagate in video - Done")
            logger.info(f"RUNTIME: propagate in video: {time.time() - start_time:.2f}[s]")
            logger.info(f"RUNTIME: Total runtime: {time.time() - start_time_all:.2f}[s]")

        return video_segments

    def box_to_segmentation(self, dl, item: su_dl.Item, annotations, progress: su_dl.Progress = None) -> list:
        return self.sam_predict_box(
            dl=dl, item=item, annotations=annotations, return_type='Semantic', progress=progress
        )

    def box_to_polygon(self, dl, item: su_dl.Item, annotations, progress: su_dl.Progress = None) -> list:
        return self.sam_predict_box(dl=dl, item=item, annotations=annotations, return_type='Polygon', progress=progress)

    def sam_predict_box(
        self, dl, item: su_dl.Item, annotations, return_type: str = 'segment', progress: su_dl.Progress = None
    ) -> list:
        """

        :param dl: DTLPY sdk instance
        :param item:
        :param annotations:
        :param return_type:
        :param progress:
        :return:
        """
        free, total, used = self.get_gpu_memory()
        print(f'Before inference GPU memory - total: {total}, used: {used}, free: {free}')

        user_email = dl.info()['user_email']
        if 'bot.dataloop.ai' in user_email and user_email != self.service_entity.bot:
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
            masks, _, _ = self.predictor.predict(
                image_properties=image_params.dict(), box=input_box, multimask_output=False
            )
            toc_model = time.time()
            logger.info(f'time to get predicted mask: {round(toc_model - tic_model, 2)} seconds')

            #######################################
            mask = masks[0]
            model_info = {'name': 'sam2', 'confidence': 1.0}

            ##########
            # Upload #
            ##########

            if return_type in ['binary', 'Semantic']:
                annotation_definition = dl.Segmentation(
                    geo=mask, label=annotation.label, attributes=annotation.attributes
                )
            elif return_type in ['segment', 'Polygon']:
                annotation_definition = dl.Polygon.from_segmentation(
                    mask=mask, label=annotation.label, attributes=annotation.attributes
                )
            else:
                raise ValueError('Unknown return type: {}'.format(return_type))
            builder = item.annotations.builder()
            builder.add(
                annotation_definition=annotation_definition,
                automated=True,
                model_info=model_info,
                metadata=annotation.metadata,
            )
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
        free, total, used = self.get_gpu_memory()
        print(f'Before inference GPU memory - total: {total}, used: {used}, free: {free}')

        user_email = dl.info()['user_email']
        if 'bot.dataloop.ai' in user_email and user_email != self.service_entity.bot:
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
            input_box = np.array([left, top, right, bottom])
            if left == right or top == bottom:
                raise ValueError(f'Bounding box too small, skipping, bbox Dimensions: {input_box}')
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

        masks, _, _ = self.predictor.predict(
            image_properties=image_params.dict(),
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
        boxed_mask = masks[0][input_box[1] : input_box[3], input_box[0] : input_box[2]]
        builder.add(annotation_definition=dl.Segmentation(geo=boxed_mask > 0, label='dummy'))
        toc_final = time.time()
        logger.info(f'time to create annotations: {round(toc_final - tic_3, 2)} seconds')
        logger.info(f'Total time of execution: {round(toc_final - tic_1, 2)} seconds')
        results = builder.annotations[0].annotation_definition.to_coordinates(color=color)
        return results


def test_local():
    import dtlpy as dl

    dl.setenv('rc')
    # dl.logout()
    # dl.login()
    item = dl.items.get(item_id='64f597eda91635a454134c79')
    item.open_in_web()

    self = Runner(dl=dl)

    ex = dl.executions.get(execution_id='693edd1547cc64790be9380e')
    ex.input['dl'] = dl
    if 'item' in ex.input:
        ex.input['item'] = dl.items.get(item_id=ex.input['item']['item_id'])
    func_to_run = getattr(self, ex.function_name)
    print(ex.input)
    results = func_to_run(**ex.input)
    builder = item.annotations.builder()
    if ex.function_name == 'track':
        result_keys = list(results.keys())
        for result_key in result_keys:
            for frame_num, coords in results[result_key].items():
                builder.add(annotation_definition=dl.Box(left=coords[0]['x'], top=coords[0]['y'], right=coords[1]['x'], bottom=coords[1]['y'], label='dummy'), frame_num=frame_num, object_id='1')
    item.annotations.upload(builder)


    print(results)

if __name__ == '__main__':
    test_local()
