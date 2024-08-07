import urllib.request
import logging
import torch
import os
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from adapters.base_grounded_sam_adapter import GroundedSamBase
import dtlpy as dl

logger = logging.getLogger('GroundedSAM2Adapter')


@dl.Package.decorators.module(description='Grounded SAM model adapter',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class GroundedSam2(GroundedSamBase):
    def load(self, local_path, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_dino(device)
        ############
        # Load SAM #
        ############
        checkpoint_filepath = self.configuration.get('checkpoint_filepath', "artifacts/sam2_hiera_large.pt")
        checkpoint_url = self.configuration.get('checkpoint_url',
                                                "https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_large.pt")
        model_cfg = os.path.join(os.getcwd(), "sam2_hiera_l.yaml")
        if not os.path.isfile(checkpoint_filepath):
            os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(checkpoint_url, checkpoint_filepath)

        # Building SAM Model and SAM Predictor
        sam2_model = build_sam2(model_cfg, checkpoint_filepath, device=device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)


if __name__ == '__main__':
    dl.setenv('rc')
    item = dl.items.get(item_id='648e90d94607411277972ad7')
    model = dl.models.get(model_id='66b33a35002458f33db87350')
    model.name = 'grounded_sam2_adapter'
    model.configuration = {}
    adapter = GroundedSam2(model_entity=model)
    adapter.predict_items(items=[item])
