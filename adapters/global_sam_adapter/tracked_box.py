import numpy as np
from adapters.global_sam_adapter.sam2_handler import DataloopSamPredictor
from sam2.modeling.sam2_base import SAM2Base
import tqdm
import cv2
import threading
from PIL import Image
import torch

def _load_img_as_tensor(img_pil, image_size):
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width
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

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        ret, frame = self.cap.read()
        img, video_height, video_width = _load_img_as_tensor(
            Image.fromarray(frame), self.image_size
        )
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


class Bbox:
    def __init__(self, x, y, w, h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    @classmethod
    def from_xyxy(cls, x1, y1, x2, y2):
        return cls(x1, y1, x2 - x1, y2 - y1)

    @classmethod
    def from_mask(cls, mask):
        if not np.any(mask):
            return None  # No mask found

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return cls.from_xyxy(xmin, ymin, xmax, ymax)

    def get_xywh(self):
        return [self.x, self.y, self.w, self.h]

    def get_xyxy(self):
        return [self.x, self.y, self.x2, self.y2]

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    @property
    def area(self):
        return self.w * self.h

    def iou(self, other):
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x2, other.x2)
        y_bottom = min(self.y2, other.y2)
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = self.w * self.h
        bb2_area = other.w * other.h
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


class TrackedBox:
    def __init__(self, bbox: Bbox, *, max_age=None):
        self.max_age = max_age
        self.bbox = bbox
        self.visible = True
        self.dead_frames = 0

    @property
    def gone(self):
        return self.max_age and self.dead_frames > self.max_age

    def track(self, sam: DataloopSamPredictor, image_params: dict, min_area=None, thresh=None):
        """
        Updates the box to the new location
        :param sam: SAM to use for tracking
        :param frame: frame to track on. if None, sam is assumed to be set with the frame (set_image)
        :param min_area:
        :param thresh:
        :return: the new bbox
        """
        if self.gone:
            return
        input_box = np.array([self.bbox.x, self.bbox.y, self.bbox.x2, self.bbox.y2])
        mask, _, _ = sam.predict(image_properties=image_params,
                                 box=input_box,
                                 multimask_output=False)
        ann_bbox = Bbox.from_mask(mask)
        df_ratio = 1 if not self.max_age else (self.max_age - self.dead_frames) / self.max_age

        if ann_bbox:  # if there is an annotation
            if not min_area or ann_bbox.area > min_area:  # and if it's bigger than min_area
                # check if the result bbox is likely correct by averaging:
                #   1. IOU with previous bbox
                #   2. confidence of the annotation
                #   3. a ratio that represents the number of dead frames (frames without detection)
                if not thresh or sum((self.bbox.iou(ann_bbox), score, df_ratio)) / 3 > thresh:
                    # if it's likely the correct bbox, update it and return.
                    self.bbox = ann_bbox
                    self.dead_frames = 0
                    self.visible = True
                    return self.bbox

        self.dead_frames += 1
        self.visible = False
