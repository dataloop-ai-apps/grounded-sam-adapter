import dtlpy as dl
from adapters.sam_adapter import SegmentAnythingAdapter


def test_local_predict_item():
    # item = dl.items.get(item_id='6493bdb47614d538df036ae9')
    item = dl.items.get(item_id='646e04519cabfb4fc7e61500')
    model_entity = dl.models.get(model_id='64c0dfd3319ec44dd2866635')
    gs = SegmentAnythingAdapter(model_entity=model_entity)
    gs.configuration['output_type'] = 'box'
    gs.configuration['points_per_side'] = 4
    annotations = gs.predict_items([item])


# def test_remote_adapter():
#     model_entity = dl.models.get(model_id='6493be8188a93d01b79c1d9c')
#     adapter = model_entity.package.build(module_name='model-adapter', init_inputs={'model_entity': model_entity})
#     item = dl.items.get(item_id='6493bdb47614d538df036ae9')
#     adapter.configuration['classes'] = ['tide', 'orange']
#     adapter.configuration['with_nms'] = False
#     adapter.configuration['box_threshold'] = 0.1
#     adapter.configuration['text_threshold'] = 0.1
#     adapter.configuration['min_area'] = 100
#     adapter.configuration['max_area'] = 10000
#     annotations = adapter.predict_items([item])


if __name__ == "__main__":
    ...
    dl.setenv('prod')
    test_local_predict_item()
