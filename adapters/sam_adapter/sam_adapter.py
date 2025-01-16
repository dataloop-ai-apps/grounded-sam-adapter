import dtlpy as dl
from adapters.sam_adapter.sam2_handler import DataloopSamPredictor
from sam2.build_sam import build_sam2
import torch
import logging
import os
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw
import base64
import io
from torch.utils.data import Dataset, DataLoader
import glob
import json

logger = logging.getLogger('[SAM2-Adapter]')


class SAMDataset(Dataset):
    def __init__(self, data_dir, training_type):
        """
        :param data_dir: Path to the folder containing downloaded images and annotations.
        :param training_type: 'point' or 'box' for different prompt types.
        """
        self.data_dir = data_dir
        self.training_type = training_type

        # Load all image paths
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpeg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True))
        # Load corresponding mask annotations
        self.annotation_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self._preprocess_image(image)

        # Load and parse annotations
        with open(annotation_path, 'r') as f:
            annotation_data = json.load(f)
        item_id = annotation_data.get('id', 'unknown_id')

        masks = self._load_masks_from_annotation(annotation_data, image.shape[:2], item_id)

        # Extract prompts
        points, point_labels = [], []
        boxes, box_labels = [], []

        if 'point' in self.training_type:
            points, point_labels = self._extract_points_from_masks(masks)

        if 'box' in self.training_type:
            boxes, box_labels = self._extract_boxes_from_annotation(annotation_data)

        # Combine prompts
        resized_masks = self._resize_and_pad_masks(masks, image.shape[:2])

        return (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(resized_masks, dtype=torch.float32),
            torch.tensor(boxes, dtype=torch.float32) if 'box' in self.training_type else None,
            torch.tensor(points, dtype=torch.float32) if 'point' in self.training_type else None,
            torch.tensor(point_labels, dtype=torch.float32) if 'point' in self.training_type else None,
            item_id
        )

    def _preprocess_image(self, image):
        # Resize and pad image to 1024x1024
        r = min(1024 / image.shape[1], 1024 / image.shape[0])
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))

        if image.shape[0] < 1024:
            image = np.pad(image, ((0, 1024 - image.shape[0]), (0, 0), (0, 0)), mode="constant")
        if image.shape[1] < 1024:
            image = np.pad(image, ((0, 0), (0, 1024 - image.shape[1]), (0, 0)), mode="constant")

        return image

    def _load_masks_from_annotation(self, annotation_data, image_size, item_id):
        masks = []
        if not annotation_data.get('annotations'):
            logger.warning(f"[Item ID: {item_id}] No annotations found in the file. Skipping this image.")
            return []
        for annotation in annotation_data['annotations']:
            if annotation['type'] == 'binary':
                encoded_data = annotation['coordinates'].split(",")[1]  # Remove prefix
                binary_data = base64.b64decode(encoded_data)
                mask = Image.open(io.BytesIO(binary_data)).convert("L")
                mask = mask.resize(image_size[::-1])  # Ensure size matches the image
                mask = np.array(mask) > 0
                masks.append(mask)

            elif annotation['type'] == 'segment':
                polygon_points = [(p['x'], p['y']) for p in annotation['coordinates'][0]]
                polygon_mask = Image.new("L", image_size, 0)
                ImageDraw.Draw(polygon_mask).polygon(polygon_points, outline=1, fill=1)
                masks.append(np.array(polygon_mask).astype(np.uint8))

        return masks

    def _extract_points_from_masks(self, masks):
        points, labels = [], []
        for mask in masks:
            coords = np.argwhere(mask)
            if coords.size > 0:
                random_coord = coords[np.random.randint(len(coords))]
                points.append([random_coord[1], random_coord[0]])  # (x, y)
                labels.append(1)
        return points, labels

    def _extract_boxes_from_annotation(self, annotation_data):
        boxes, labels = [], []
        for annotation in annotation_data['annotations']:
            if annotation['type'] == 'box':
                coords = annotation['coordinates']
                x_min, y_min = int(coords[0]['x']), int(coords[0]['y'])
                x_max, y_max = int(coords[1]['x']), int(coords[1]['y'])
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)
        return boxes, labels

    def _resize_and_pad_masks(self, masks, image_size):
        resized_masks = []
        r = min(1024 / image_size[0], 1024 / image_size[1])
        for mask in masks:
            resized_mask = cv2.resize(mask.astype(np.uint8), (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                                      interpolation=cv2.INTER_NEAREST)
            if resized_mask.shape[0] < 1024:
                resized_mask = np.pad(resized_mask, ((0, 1024 - resized_mask.shape[0]), (0, 0)), mode="constant")
            if resized_mask.shape[1] < 1024:
                resized_mask = np.pad(resized_mask, ((0, 0), (0, 1024 - resized_mask.shape[1])), mode="constant")
            resized_masks.append(resized_mask)
        return resized_masks


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

        sam2_model = build_sam2(model_cfg, weights_filepath, device=self.device)
        self.predictor = DataloopSamPredictor(sam2_model)
        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        if self.configuration.get('was_trained', 'False') is True:
            checkpoint_path = os.path.join(local_path, "best_sam2_model.torch")
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.predictor.model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
            logger.info(f"Model loaded from {checkpoint_path}.")
        return

    def prepare_item_func(self, item):
        return item

    def predict(self, batch, **kwargs):
        logger.info("Running SAM2 predictions")
        batch_annotations = list()
        for item in batch:
            image_annotations = dl.AnnotationCollection()
            item_annotations = item.annotations.list()
            image = item.download(save_locally=False, to_array=True, overwrite=True)
            image_properties = self.predictor.set_image(image=image)
            if not item_annotations.annotations:
                image_width = item.width
                image_height = item.height
                input_box = np.array([0, 0, image_width, image_height])
                masks, _, _ = self.predictor.predict(image_properties=image_properties,
                                                     box=input_box,
                                                     multimask_output=False)

                mask = masks[0]
                # annotation_definition = dl.Segmentation(geo=mask, label=annotation.label)
                annotation_definition = dl.Polygon.from_segmentation(mask=mask, label="NA")

                image_annotations.add(annotation_definition=annotation_definition,
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': 1})
            else:
                for annotation in item_annotations:
                    if annotation.type != "box":
                        continue
                    coordinates = annotation.coordinates
                    left = int(coordinates[0]['x'])
                    top = int(coordinates[0]['y'])
                    right = int(coordinates[1]['x'])
                    bottom = int(coordinates[1]['y'])
                    input_box = np.array([left, top, right, bottom])
                    masks, _, _ = self.predictor.predict(image_properties=image_properties,
                                                         box=input_box,
                                                         multimask_output=False)

                    mask = masks[0]
                    # annotation_definition = dl.Segmentation(geo=mask, label=annotation.label)
                    annotation_definition = dl.Polygon.from_segmentation(mask=mask, label=annotation.label,
                                                                         attributes=annotation.attributes)

                    image_annotations.add(annotation_definition=annotation_definition,
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': 1},
                                          metadata=annotation.metadata)
            batch_annotations.append(image_annotations)
        logger.info("Finished running SAM2 predictions")
        return batch_annotations

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param data_path: `str` local File System directory path where
                           we already downloaded the data from dataloop platform
        :return:
        """

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_model which first save locally and then uploads to model entity

        :param local_path: `str` directory path in local FileSystem
        """
        torch.save(self.predictor.model.state_dict(), os.path.join(local_path, "model.torch"))
        self.configuration['state_dict'] = 'model.torch'
        self.configuration['was_trained'] = True

    def set_train_mode(self):
        """
        Set only the mask decoder and prompt encoder to training mode.
        Leave the image encoder untouched.
        """
        self.predictor.model.sam_mask_decoder.train()
        self.predictor.model.sam_prompt_encoder.train()

    def set_eval_mode(self):
        """
        Set only the mask decoder and prompt encoder to evaluation mode.
        Leave the image encoder untouched.
        """
        self.predictor.model.sam_mask_decoder.eval()
        self.predictor.model.sam_prompt_encoder.eval()

    def _decode_masks(self, sparse_embeddings, dense_embeddings):
        """
        Decode masks using the SAM model.

        :param sparse_embeddings: Sparse embeddings from the prompt encoder.
        :param dense_embeddings: Dense embeddings from the prompt encoder.
        :return: Low-resolution masks and predicted scores.
        """
        high_res_features = [
            feat_level[-1].unsqueeze(0) for feat_level in self.predictor._features["high_res_feats"]
        ]

        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks, prd_scores

    def _prepare_prompts(self, image_properties, boxes=None, points=None, point_labels=None, training_type=None):
        """
        Prepare prompts (points and/or boxes) for the SAM model.

        :param image_properties: Image properties set by the predictor.
        :param boxes: Bounding box prompts.
        :param points: Point prompts.
        :param point_labels: Labels for point prompts.
        :param training_type: List specifying the training type(s): ["point"], ["box"], or ["point", "box"].
        :return: mask_input, unnorm_coords, input_labels, embeddings
        """

        # Initialize empty prompts if missing
        if boxes is None or len(boxes) == 0:
            boxes = None
        if points is None or len(points) == 0:
            points = None
            point_labels = None

        # Prepare combined prompts
        return self.predictor._prep_prompts(
            image_properties=image_properties,
            point_coords=points,
            point_labels=point_labels,
            box=boxes,
            mask_logits=None,
            normalize_coords=True,
        )

    def _process_epoch_dataloader(self, dataloader, training_type, scaler, optimizer, mode="train"):
        total_loss = 0.0
        num_batches = 0

        self.set_train_mode() if mode == "train" else self.set_eval_mode()

        for batch in dataloader:
            images, masks, boxes, points, point_labels, item_ids = batch

            images = images.to(self.device)
            masks = masks.to(self.device)

            if "box" in training_type:
                boxes = boxes.to(self.device)

            if "point" in training_type:
                points = points.to(self.device)
                point_labels = point_labels.to(self.device)

            context = torch.no_grad() if mode == "validate" else torch.enable_grad()

            with context:
                with torch.cuda.amp.autocast():
                    batch_loss = 0.0
                    valid_images = 0

                    for i, image in enumerate(images):
                        item_id = item_ids[i]

                        image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        image_properties = self.predictor.set_image(image=image_np)

                        # Skip if "point" but no mask
                        if "point" in training_type and masks[i].sum() == 0:
                            logger.warning(
                                f"[Item ID: {item_id}] "
                                f"No valid mask found in point-based training. Skipping this image.")
                            continue

                        # Skip if "box" but no box annotation
                        if "box" in training_type and (boxes is None or boxes[i].sum() == 0):
                            logger.warning(
                                f"[Item ID: {item_id}] "
                                f"Box-based training specified but no box annotation found. Continuing without box.")

                        # Check if both prompts are missing
                        box_prompt = boxes[i] if "box" in training_type and boxes[i].sum() > 0 else None
                        point_prompt = points[i] if "point" in training_type and points[i].sum() > 0 else None

                        if "point" in training_type and "box" in training_type and (
                                box_prompt is None and point_prompt is None):
                            logger.warning(
                                f"[Item ID: {item_id}] Both box and point prompts are missing. Skipping this image.")
                            continue

                        # Prepare prompts
                        mask_input, unnorm_coords, input_labels, _ = self._prepare_prompts(
                            image_properties=image_properties,
                            boxes=box_prompt,
                            points=point_prompt,
                            point_labels=point_labels[i] if "point" in training_type else None,
                            training_type=training_type
                        )

                        # Generate embeddings
                        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                            points=(unnorm_coords, input_labels) if point_prompt is not None else None,
                            boxes=box_prompt,
                            masks=None,
                        )

                        # Decode masks
                        low_res_masks, prd_scores = self._decode_masks(sparse_embeddings, dense_embeddings)
                        prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks,
                                                                                 self.predictor._orig_hw[-1])
                        prd_probs = torch.sigmoid(prd_masks[:, 0])

                        # Segmentation loss
                        seg_loss = (-masks[i] * torch.log(prd_probs + 1e-7) -
                                    (1 - masks[i]) * torch.log(1 - prd_probs + 1e-7)).mean()

                        # IoU-based score loss
                        inter = (masks[i] * (prd_probs > 0.5)).sum()
                        union = masks[i].sum() + (prd_probs > 0.5).sum() - inter + 1e-7
                        iou = inter / union
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                        # Total loss
                        total_batch_loss = seg_loss + 0.05 * score_loss

                        # Training step
                        if mode == "train":
                            optimizer.zero_grad()
                            scaler.scale(total_batch_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        batch_loss += total_batch_loss.item()
                        valid_images += 1

                    if valid_images > 0:
                        total_loss += batch_loss / valid_images
                        num_batches += 1

        return total_loss / max(num_batches, 1)

    def train(self, data_path, output_path, **kwargs):
        num_epochs = self.configuration.get('num_epochs', 100)
        learning_rate = self.configuration.get('learning_rate', 1e-5)
        weight_decay = self.configuration.get('weight_decay', 4e-5)
        batch_size = self.configuration.get('batch_size', 4)
        save_interval = 5
        patience = 3
        no_improvement_epochs = 0

        train_path = os.path.join(data_path, 'train')
        validation_path = os.path.join(data_path, 'validation')
        self.model_entity.dataset.download(
            filters=dl.Filters(custom_filter=self.model_entity.metadata['system']['subsets']['train']['filter']),
            local_path=train_path,
            overwrite=True,
            annotation_options=['json']
        )
        self.model_entity.dataset.download(
            filters=dl.Filters(custom_filter=self.model_entity.metadata['system']['subsets']['validation']['filter']),
            local_path=validation_path,
            overwrite=True,
            annotation_options=['json']
        )

        training_type = self.configuration.get('training_type', ["point", "box"])
        train_dataset = SAMDataset(train_path, training_type)
        val_dataset = SAMDataset(validation_path, training_type)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        self.set_train_mode()
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss = self._process_epoch_dataloader(train_loader, training_type, scaler, optimizer, mode="train")
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

            val_loss = self._process_epoch_dataloader(val_loader, training_type, scaler, optimizer, mode="validate")
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                torch.save(self.predictor.model.state_dict(), f"{output_path}/sam2_model_epoch_{epoch + 1}.torch")
                logger.info(f"Checkpoint saved at epoch {epoch + 1}.")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                torch.save(self.predictor.model.state_dict(), f"{output_path}/best_sam2_model.torch")
                logger.info("Best model saved.")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
