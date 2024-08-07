import urllib.request
import logging
import torch
import os
from segment_anything import sam_model_registry, SamPredictor
from adapters.base_grounded_sam_adapter import GroundedSamBase
import dtlpy as dl

logger = logging.getLogger('GroundedAdapter')


@dl.Package.decorators.module(description='Grounded SAM model adapter',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class GroundedSam(GroundedSamBase):
    def load(self, local_path, **kwargs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_dino(device)
        ############
        # Load SAM #
        ############
        sam_weights_url = 'https://storage.googleapis.com/model-mgmt-snapshots/sam/sam_vit_b_01ec64.pth'
        sam_checkpoint_filepath = self.configuration.get('sam_checkpoint_filepath', "artifacts/sam_vit_b_01ec64.pth")
        sam_checkpoint_url = self.configuration.get('sam_checkpoint_url', sam_weights_url)
        sam_model_type = self.configuration.get('sam_model_type', "vit_b")

        if not os.path.isfile(sam_checkpoint_filepath):
            os.makedirs(os.path.dirname(sam_checkpoint_filepath), exist_ok=True)
            urllib.request.urlretrieve(sam_checkpoint_url, sam_checkpoint_filepath)
        logger.info(f'loading weights sam_checkpoint_path: {sam_checkpoint_filepath}')
        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_filepath)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)


if __name__ == '__main__':
    dl.setenv('rc')
    item = dl.items.get(item_id='66b352dc1d2d6c6530447c04')
    model = dl.models.get(model_id='66b33a35002458f33db87350')
    model.name = 'grounded_sam_adapter'
    model.configuration = {}
    adapter = GroundedSam(model_entity=model)
    adapter.predict_items(items=[item])
