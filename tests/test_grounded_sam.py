import dtlpy as dl
from adapters.grounded_sam_adapter.grounded_sam_adapter import GroundedSam
from PIL import Image
import numpy as np


def test_predict_local():
    gs = GroundedSam()
    gs.load(None)
    image = Image.open('../assets/000000001296.jpg')
    annotations = gs.predict([np.asarray(image)],
                             save_results=True,
                             classes=['person', 'cellphone'])


def test_local_predict_item():
    # item = dl.items.get(item_id='6493bdb47614d538df036ae9')
    # item = dl.items.get(item_id='649bf3fae8675cdd09bec43a')
    item = dl.items.get(item_id='64c6b7994f51dc7cb8f70492')
    model_entity = dl.models.get(model_id='649af4bfeced7a8c694564b9')
    gs = GroundedSam(model_entity=model_entity)
    gs.configuration['classes'] = {'pepper': {'min_area': 100}, 'crack': {'min_area': 100}}
    # gs.configuration['classes'] = {'pepper': {'min_area': 100}}
    # gs.configuration['with_nms'] = False
    # gs.configuration['box_threshold'] = 0.1
    # gs.configuration['text_threshold'] = 0.1

    annotations = gs.predict_items([item])


def test_remote_adapter():
    model_entity = dl.models.get(model_id='6493be8188a93d01b79c1d9c')
    adapter = model_entity.package.build(module_name='model-adapter', init_inputs={'model_entity': model_entity})
    item = dl.items.get(item_id='6493bdb47614d538df036ae9')
    adapter.configuration['classes'] = ['tide', 'orange']
    adapter.configuration['with_nms'] = False
    adapter.configuration['box_threshold'] = 0.1
    adapter.configuration['text_threshold'] = 0.1
    adapter.configuration['min_area'] = 100
    adapter.configuration['max_area'] = 10000
    annotations = adapter.predict_items([item])


if __name__ == "__main__":
    ...
    dl.setenv('prod')
    test_local_predict_item()
