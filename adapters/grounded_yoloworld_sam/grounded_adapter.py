import dtlpy as dl
from adapters.grounded_yoloworld_sam.sam2_adapter import Runner as SAMAdapter
from adapters.grounded_yoloworld_sam.yoloworld_adapter import YOLOWorldAdapter
import logging

logger = logging.getLogger('Grounded-YoloWorld-Sam2-Adapter')


class GroundedYOLOSAMAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        self.yolo_adapter = None
        self.sam2_adapter = None
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        """
        Load the wrapped YOLOWorld adapter and the SAM2 service.
        """
        # Initialize YOLOWorldAdapter
        self.yolo_adapter = YOLOWorldAdapter(model_entity=self.model_entity)
        logger.info("YOLOWorld Adapter successfully loaded")

        # Connect to SAM2 Service
        self.sam2_adapter = SAMAdapter()
        logger.info(f"SAM2 Adapter successfully loaded")

    def prepare_item_func(self, item):
        return item

    def get_labels(self, item):
        return [label.tag for label in item.dataset.labels]

    def predict(self, batch, **kwargs):
        """
        Perform predictions using YOLOWorld and pass bounding boxes to SAM2.
        """
        dataset_labels = self.get_labels(batch[0])
        logger.info(f"Running YOLOWorld predictions on dataset labels : {dataset_labels}")
        self.yolo_adapter.model.set_classes(dataset_labels)
        images = [self.yolo_adapter.prepare_item_func(item) for item in batch]
        yolo_predictions = self.yolo_adapter.predict(images, **kwargs)
        logger.info("Finished running YOLOWorld predictions")

        logger.info("Running SAM2 predictions")
        batch_annotations = list()
        for item, annotations in zip(batch, yolo_predictions):
            image_annotations = dl.AnnotationCollection()
            bounding_boxes = [annotation.to_json() for annotation in annotations]
            if not bounding_boxes:
                continue
            sam2_prediction = self.sam2_adapter.sam_predict_box(item=item, annotations=bounding_boxes)

            sam_ann = dl.Annotation.from_json(sam2_prediction[0])
            image_annotations.add(annotation_definition=dl.Polygon(geo=sam_ann.geo, label=sam_ann.label),
                                  model_info={'name': self.model_entity.name,
                                              'model_id': self.model_entity.id,
                                              'confidence': 1},
                                  metadata=annotations[0].metadata)
            batch_annotations.append(image_annotations)
        logger.info("Finished running SAM2 predictions")
        return batch_annotations
