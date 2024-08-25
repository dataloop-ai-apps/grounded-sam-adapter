import os.path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import logging
import urllib
import dtlpy as dl
import torch
from adapters.base_sam_adapter import SAMAdapter

logger = logging.getLogger('Sam2Adapter')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_WIDTH = MAX_HEIGHT = 640


@dl.Package.decorators.module(description='Model Adapter for Segment Anything',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class SegmentAnything2Adapter(SAMAdapter):

    def load(self, local_path, **kwargs):
        logger.info(f'loading model weights. device: {DEVICE}')
        checkpoint_filepath = self.configuration.get('checkpoint_filepath', "artifacts/sam2_hiera_large.pt")
        checkpoint_url = self.configuration.get('checkpoint_url',
                                                "https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_large.pt")
        model_cfg = os.path.join(os.getcwd(), "sam2_hiera_l.yaml")

        if not os.path.isfile(checkpoint_filepath):
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint_filepath)

        logger.info("Finished downloading model weights")
        # SAM2 Options
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
        logger.info('Initializing model')
        sam2 = build_sam2(model_cfg, checkpoint_filepath, apply_postprocessing=False)
        self.model = SAM2AutomaticMaskGenerator(model=sam2,
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
                                                min_mask_region_area=min_mask_region_area)


if __name__ == '__main__':
    dl.setenv('rc')
    item = dl.items.get(item_id='648e90d94607411277972ad7')
    model = dl.models.get(model_id='66b33a35002458f33db87350')
    model.name = 'sam2_adapter'
    model.configuration = {}
    adapter = SegmentAnything2Adapter(model_entity=model)
    adapter.predict_items(items=[item])
