import logging
import os
import urllib
import dtlpy as dl
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from adapters.base_sam_adapter import SAMAdapter

logger = logging.getLogger('SamAdapter')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WIDTH = MAX_HEIGHT = 640


@dl.Package.decorators.module(description='Model Adapter for Segment Anything',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class SegmentAnythingAdapter(SAMAdapter):

    def load(self, local_path, **kwargs):
        logger.info(f'loading model weights. device: {DEVICE}')
        checkpoint_filepath = self.configuration.get('checkpoint_filepath', "artifacts/sam_vit_b_01ec64.pth")
        checkpoint_url = self.configuration.get('checkpoint_url',
                                                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        model_type = self.configuration.get('model_type', "vit_b")

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


if __name__ == '__main__':
    dl.setenv('rc')
    item = dl.items.get(item_id='66b352dc1d2d6c6530447c04')
    model = dl.models.get(model_id='66b33a35002458f33db87350')
    model.name = 'sam_adapter'
    model.configuration = {}
    adapter = SegmentAnythingAdapter(model_entity=model)
    adapter.predict_items(items=[item])
