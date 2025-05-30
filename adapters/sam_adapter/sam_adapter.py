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
from torch.utils.data import Dataset
import glob
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import base64
from PIL import Image, ImageDraw
import io

logger = logging.getLogger('[SAM2-Adapter]')
logging.getLogger('PIL').setLevel(logging.WARNING)


class SAMDataset(Dataset):
    def __init__(self, data_dir):
        """
        :param data_dir: Path to the folder containing images and annotations.
        """
        self.data_dir = data_dir

        # Load all image paths
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.jpeg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True) +
                                  glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True))
        # Load corresponding annotations
        self.annotation_paths = sorted(glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True))

        # Create expanded dataset entries that account for polygons and masks
        self.dataset_entries = []
        for img_path, ann_path in zip(self.image_paths, self.annotation_paths):
            with open(ann_path, 'r') as f:
                annotation_data = json.load(f)

            # Group annotations by type
            binary_annotations = None
            polygon_annotations = []

            if annotation_data.get('annotations'):
                for ann in annotation_data['annotations']:
                    if ann['type'] == 'binary':
                        binary_annotations = ann
                    elif ann['type'] == 'segment':
                        polygon_annotations.append(ann)
            else:
                logger.warning(
                    f"No annotations found for item {annotation_data.get('id', 'unknown_id')}. Skipping item.")
                continue

            # If we have binary annotations, create one entry with the mask
            if binary_annotations:
                self.dataset_entries.append({
                    'image_path': img_path,
                    'annotation': binary_annotations,
                    'item_id': annotation_data.get('id', 'unknown_id')
                })

            # For each polygon, create a separate entry
            for poly_ann in polygon_annotations:
                self.dataset_entries.append({
                    'image_path': img_path,
                    'annotation': poly_ann,
                    'item_id': annotation_data.get('id', 'unknown_id')
                })

    def __len__(self):
        return len(self.dataset_entries)

    def __getitem__(self, idx):
        entry = self.dataset_entries[idx]
        image_path = entry['image_path']
        annotation = entry['annotation']
        item_id = entry['item_id']

        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_height, original_width = image.shape[:2]
        image = self._preprocess_image(image, (1024, 1024))

        # Generate mask with unique labels
        mask = self._load_from_annotation(annotation, original_width, original_height)
        if mask is None:
            logger.warning(f"No mask found for item {item_id}. Skipping item.")
            return None, None, None, None, item_id

        # Generate points from the mask
        points, point_labels = self._extract_points_from_mask(mask)

        # Prepare binary mask for PyTorch
        binary_mask = np.expand_dims(mask, axis=-1)
        binary_mask = binary_mask.transpose((2, 0, 1))

        points = np.expand_dims(points, axis=1)
        point_labels = np.expand_dims(point_labels, axis=1)

        return (
            torch.tensor(image, dtype=torch.float32).permute(2, 0, 1),
            torch.tensor(binary_mask, dtype=torch.float32),
            torch.tensor(points, dtype=torch.float32),
            torch.tensor(point_labels, dtype=torch.float32),
            item_id
        )

    def _pad_to_size(self, array, target_size, pad_value=0):
        """
        Pad array to target size with equal padding on both sides.
        
        Args:
            array: Input array to pad (can be image or mask)
            target_size: Tuple of (height, width) for target size
            pad_value: Value to use for padding (0 for masks, (0,0,0) for RGB images)
            
        Returns:
            Padded array of target size
        """
        if len(array.shape) == 3:  # For RGB images
            h, w, c = array.shape
        else:  # For masks
            h, w = array.shape

        delta_h = target_size[0] - h
        delta_w = target_size[1] - w

        top = delta_h // 2
        bottom = delta_h - top
        left = delta_w // 2
        right = delta_w - left

        if len(array.shape) == 3:
            return cv2.copyMakeBorder(array, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)
        else:
            return np.pad(array, ((top, bottom), (left, right)), mode="constant", constant_values=pad_value)

    def _preprocess_image(self, image, target_size):
        """
        Resize and pad the image to the target size.
        """
        h, w = image.shape[:2]
        r = min(target_size[1] / w, target_size[0] / h)
        new_w, new_h = int(w * r), int(h * r)
        resized_image = cv2.resize(image, (new_w, new_h))
        padded_image = self._pad_to_size(resized_image, target_size, (0, 0, 0))
        return padded_image

    def _load_from_annotation(self, annotation, original_width, original_height):
        """
        Combines all masks into a single mask with unique labels for each region and retrieves boxes annotations.
        """
        target_size = (1024, 1024)
        scaling_factor = min(target_size[1] / original_width, target_size[0] / original_height)
        new_width, new_height = int(original_width * scaling_factor), int(original_height * scaling_factor)

        mask = np.zeros(target_size, dtype=np.uint8)
        if annotation['type'] == 'binary':
            encoded_data = annotation['coordinates'].split(",")[1]
            binary_data = base64.b64decode(encoded_data)
            mask = Image.open(io.BytesIO(binary_data)).convert("L")
            mask = mask.resize((new_width, new_height), Image.NEAREST)
            mask = np.array(mask) > 0
        elif annotation['type'] == 'segment':
            polygon_points = [(int(p['x'] * scaling_factor), int(p['y'] * scaling_factor))
                              for p in annotation['coordinates'][0]]
            polygon_mask = Image.new("L", (new_width, new_height), 0)
            ImageDraw.Draw(polygon_mask).polygon(polygon_points, outline=1, fill=1)
            mask = np.array(polygon_mask).astype(bool)
        else:
            logger.warning(f"Annotation type {annotation['type']} not supported")
            return mask
        padded_mask = self._pad_to_size(mask, target_size, 0)

        return padded_mask

    def _extract_points_from_mask(self, mask):
        """
        Extract 3 points inside and 3 points outside the mask.
        Returns points in (x,y) format.
        """
        NUM_POINTS = 5  # Number of points to extract for each class
        points = []
        point_labels = []

        binary_mask = (mask > 0).astype(np.uint8)

        # Get inside points
        inside_coords = np.argwhere(binary_mask > 0)
        if len(inside_coords) < NUM_POINTS:
            # Randomly select points with replacement if not enough
            indices = np.random.choice(len(inside_coords), NUM_POINTS, replace=True)
        else:
            # Randomly select points without replacement
            indices = np.random.choice(len(inside_coords), NUM_POINTS, replace=False)
        inside_coords = inside_coords[indices]

        # Get outside points
        outside_coords = np.argwhere(binary_mask == 0)
        if len(outside_coords) < NUM_POINTS:
            # Randomly select points with replacement if not enough
            indices = np.random.choice(len(outside_coords), NUM_POINTS, replace=True)
        else:
            # Randomly select points without replacement
            indices = np.random.choice(len(outside_coords), NUM_POINTS, replace=False)
        outside_coords = outside_coords[indices]

        for coord in inside_coords:
            points.append([coord[1], coord[0]])
            point_labels.append(1)
        for coord in outside_coords:
            points.append([coord[1], coord[0]])
            point_labels.append(0)

        points = np.array(points)
        point_labels = np.array(point_labels)

        return points, point_labels


class ModelAdapter(dl.BaseModelAdapter):
    def __init__(self, model_entity: dl.Model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'GPU available: {torch.cuda.is_available()}')

        super().__init__(model_entity)

    def load(self, local_path, **kwargs):
        # Get model size from config, defaulting to small
        model_cfg = self.configuration.get('model_cfg', 'sam2_hiera_s.yaml')
        model_size = 'small' if 'sam2_hiera_s' in model_cfg else 'large'

        # Construct URLs and filenames based on model size
        weights_url = f'https://storage.googleapis.com/model-mgmt-snapshots/sam2/sam2_hiera_{model_size}.pt'
        weights_filepath = os.path.join('artifacts', f'sam2_hiera_{model_size}.pt')
        if not os.path.isfile(weights_filepath):
            os.makedirs(os.path.dirname(weights_filepath), exist_ok=True)
            urllib.request.urlretrieve(weights_url, weights_filepath)

        sam2_model = build_sam2(model_cfg, weights_filepath, device=self.device)
        self.predictor = DataloopSamPredictor(sam2_model)

        self.save_model_name = self.configuration.get('save_model_name', 'best_sam2_model.torch')
        model_filename = os.path.join(local_path, self.configuration.get('save_model_name', self.save_model_name))
        if os.path.exists(model_filename):
            logger.info("Loading trained weights.")
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.predictor.model.load_state_dict(torch.load(model_filename, map_location=map_location))
        else:
            logger.info("No trained weights file found. Loading pre-trained weights.")

        self.multi_points_prediction = self.configuration.get('multi_points_prediction', False)
        self.output_type = self.configuration.get('output_type', 'segment')

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
                masks, scores, _ = self.predictor.predict(image_properties=image_properties,
                                                          box=input_box,
                                                          multimask_output=True)

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]

                for i, mask in enumerate(masks):
                    if self.output_type == "segment":
                        annotation_definition = dl.Segmentation(geo=mask, label=f"mask_{i+1}")
                    elif self.output_type == "polygon":
                        annotation_definition = dl.Polygon.from_segmentation(mask=mask, label=f"mask_{i+1}")
                    image_annotations.add(
                        annotation_definition=annotation_definition,
                        model_info={
                            'name': self.model_entity.name,
                            'model_id': self.model_entity.id,
                            'confidence': scores[i]
                        }
                    )
            elif self.multi_points_prediction:
                input_points = []
                input_labels = []
                for annotation in item_annotations:
                    coordinates = annotation.coordinates
                    if annotation.type == "point" and annotation.label == "inside":
                        input_points.append([coordinates['x'], coordinates['y']])
                        input_labels.append(1)
                    elif annotation.type == "point" and annotation.label == "outside":
                        input_points.append([coordinates['x'], coordinates['y']])
                        input_labels.append(0)
                    else:
                        logger.info(f"Annotation Type {annotation.type} not supported for multi-points prediction. Use points with labels 'inside' or 'outside'.")
                        continue

                input_points = np.array(input_points)
                input_labels = np.array(input_labels)
                
                masks, scores, _ = self.predictor.predict(image_properties=image_properties,
                                                          point_coords=input_points,
                                                          point_labels=input_labels,
                                                          multimask_output=False)
                mask = masks[0]
                if self.output_type == "segment":
                    annotation_definition = dl.Segmentation(geo=mask, label='mask')
                elif self.output_type == "polygon":
                    annotation_definition = dl.Polygon.from_segmentation(mask=mask, label='mask')
                image_annotations.add(annotation_definition=annotation_definition,
                                      model_info={'name': self.model_entity.name,
                                                  'model_id': self.model_entity.id,
                                                  'confidence': scores[0]})
            else:
                for annotation in item_annotations:
                    coordinates = annotation.coordinates
                    if annotation.type == "box":
                        left = int(coordinates[0]['x'])
                        top = int(coordinates[0]['y'])
                        right = int(coordinates[1]['x'])
                        bottom = int(coordinates[1]['y'])
                        input_box = np.array([left, top, right, bottom])
                        masks, scores, _ = self.predictor.predict(image_properties=image_properties,
                                                                  box=input_box,
                                                                  multimask_output=False)

                    elif annotation.type == "point":
                        input_point = np.array([[coordinates['x'], coordinates['y']]])
                        input_label = np.array([1])
                        masks, scores, _ = self.predictor.predict(image_properties=image_properties,
                                                                  point_coords=input_point,
                                                                  point_labels=input_label,
                                                                  multimask_output=False)
                    else:
                        logger.info(f"Annotation Type {annotation.type} not supported. Please use points or boxes.")
                        continue
                    mask = masks[0]
                    if self.output_type == "segment":
                        annotation_definition = dl.Segmentation(geo=mask, label=annotation.label)
                    elif self.output_type == "polygon":
                        annotation_definition = dl.Polygon.from_segmentation(mask=mask, label=annotation.label)
                    image_annotations.add(annotation_definition=annotation_definition,
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': scores[0]})
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
        torch.save(self.predictor.model.state_dict(), os.path.join(local_path, self.save_model_name))
        self.configuration['save_model_name'] = self.save_model_name

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
        high_res_features = self.predictor._features["high_res_feats"]

        image_embeddings = self.predictor._features["image_embed"][-1].unsqueeze(0)

        low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        return low_res_masks, prd_scores

    def _process_epoch_dataloader(self, dataloader, scaler, optimizer, mode="train"):
        total_loss = 0.0
        num_batches = 0

        self.set_train_mode() if mode == "train" else self.set_eval_mode()

        for batch in dataloader:
            images, masks, points, point_labels, item_ids = batch

            images = images.to(self.device)
            masks = masks.to(self.device)
            points = points.to(self.device)
            point_labels = point_labels.to(self.device)

            # Use no_grad for validation
            context = torch.no_grad() if mode == "validate" else torch.enable_grad()

            with context:
                with torch.cuda.amp.autocast():
                    batch_loss = 0.0
                    valid_images = 0

                    for i, image in enumerate(images):
                        mask = masks[i]
                        input_points = points[i].squeeze(1)
                        input_labels = point_labels[i].squeeze(1)
                        item_id = item_ids[i]

                        if image is None:
                            logger.warning(
                                f"[Item ID: {item_id}] No valid image found. Skipping this image.")
                            continue
                        if mask is None:
                            logger.warning(
                                f"[Item ID: {item_id}] No valid mask found. Skipping this image.")
                            continue

                        # Prepare the image for the predictor
                        image_np = (image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        image_properties = self.predictor.set_image(image=image_np)

                        # Prepare prompts
                        mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                            image_properties=image_properties,
                            point_coords=input_points,
                            point_labels=input_labels,
                            box=None,
                            mask_logits=None,
                            normalize_coords=True,
                        )

                        # Generate embeddings
                        sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                            points=(unnorm_coords, labels), boxes=None, masks=None,
                        )

                        # Decode masks
                        low_res_masks, prd_scores = self._decode_masks(sparse_embeddings, dense_embeddings)
                        prd_masks = self.predictor._transforms.postprocess_masks(
                            low_res_masks, self.predictor._orig_hw[-1]
                        )
                        prd_probs = torch.sigmoid(prd_masks[:, 0])  # Predicted probabilities

                        # Segmentation loss (binary cross-entropy)
                        seg_loss = (-mask * torch.log(prd_probs + 1e-7) -
                                    (1 - mask) * torch.log(1 - prd_probs + 1e-7)).mean()

                        # IoU-based score loss
                        inter = (mask * (prd_probs > 0.5)).sum()
                        union = mask.sum() + (prd_probs > 0.5).sum() - inter + 1e-7
                        iou = inter / union
                        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                        # Total loss
                        total_batch_loss = seg_loss + 0.05 * score_loss

                        # Backward pass and optimizer step (only in training mode)
                        if mode == "train":
                            optimizer.zero_grad()
                            scaler.scale(total_batch_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        # Accumulate loss for the batch
                        batch_loss += total_batch_loss.item()
                        valid_images += 1

                    # Average loss per valid image
                    if valid_images > 0:
                        total_loss += batch_loss / valid_images
                        num_batches += 1

        # Return average loss across all batches
        return total_loss / max(num_batches, 1)

    def train(self, data_path, output_path, **kwargs):
        num_epochs = self.configuration.get('num_epochs', 100)
        learning_rate = self.configuration.get('learning_rate', 1e-5)
        weight_decay = self.configuration.get('weight_decay', 4e-5)
        batch_size = self.configuration.get('batch_size', 4)
        save_interval = self.configuration.get('save_interval', 10)
        patience = self.configuration.get('patience', 10)

        no_improvement_epochs = 0

        train_path = os.path.join(data_path, 'train')
        validation_path = os.path.join(data_path, 'validation')

        train_dataset = SAMDataset(train_path)
        val_dataset = SAMDataset(validation_path)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')
        self.set_train_mode()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            train_loss = self._process_epoch_dataloader(train_loader, scaler, optimizer, mode="train")
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
            print(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")

            val_loss = self._process_epoch_dataloader(val_loader, scaler, optimizer, mode="validate")
            logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
            print(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                torch.save(self.predictor.model.state_dict(), f"{output_path}/sam2_training_epoch_{epoch + 1}.torch")
                logger.info(f"Checkpoint saved at epoch {epoch + 1}.")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
                torch.save(self.predictor.model.state_dict(), f"{output_path}/{self.save_model_name}")
                logger.info("Best model saved.")
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
