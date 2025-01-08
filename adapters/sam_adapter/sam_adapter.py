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

logger = logging.getLogger('[SAM2-Adapter]')


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
            self.predictor.model.load_state_dict(torch.load(os.path.join(local_path, "model.torch")))
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

    def read_batch(self, item, image_size):
        """
        Prepare the image, individual masks, and points for training.

        :param item: Dataloop item.
        :param image_size: Tuple (width, height) of the image.
        :return: Image, masks, points, and labels for the batch.
        """
        # Initialize empty mask and point containers
        masks = []
        points = []
        labels = []

        for annotation in item.annotations.list():
            # Decode SEGMENTATION annotations
            if annotation.type == dl.AnnotationType.SEGMENTATION:
                encoded_data = annotation.coordinates.split(",")[1]  # Remove "data:image/png;base64,"
                binary_data = base64.b64decode(encoded_data)
                mask = Image.open(io.BytesIO(binary_data))

                # Convert mask to binary numpy array and ensure it's 2D
                mask = np.array(mask)
                if mask.ndim == 3:  # Handle multi-channel masks
                    mask = mask[..., 0]
                mask = mask > 0  # Binary mask

                # Store the mask as a separate binary mask
                masks.append(mask)

                # Select a random point inside the mask
                coords = np.argwhere(mask > 0)
                if coords.size > 0:
                    random_coord = coords[np.random.randint(len(coords))]  # Random coordinate
                    points.append([random_coord[1], random_coord[0]])  # (x, y) format
                    labels.append(1)  # Positive point label

            # Handle POLYGON annotations
            elif annotation.type == dl.AnnotationType.POLYGON:
                polygon_points = [(point["x"], point["y"]) for point in annotation.coordinates[0]]
                polygon_mask = Image.new("L", image_size, 0)
                ImageDraw.Draw(polygon_mask).polygon(polygon_points, outline=1, fill=1)
                polygon_mask = np.array(polygon_mask).astype(np.uint8)

                # Store the mask as a separate binary mask
                masks.append(polygon_mask)

                # Select a random point inside the mask
                coords = np.argwhere(polygon_mask > 0)
                if coords.size > 0:
                    random_coord = coords[np.random.randint(len(coords))]  # Random coordinate
                    points.append([random_coord[1], random_coord[0]])  # (x, y) format
                    labels.append(1)  # Positive point label

        # Convert points and labels to numpy arrays
        points = np.array(points).reshape(-1, 1, 2)  # (num_points, 1, 2)
        labels = np.array(labels).reshape(-1, 1)  # (num_points, 1)

        # Download and preprocess the image
        image = item.download(save_locally=False, to_array=True, overwrite=True)
        r = np.min([1024 / image.shape[1], 1024 / image.shape[0]])  # Scaling factor
        image = cv2.resize(image, (int(image.shape[1] * r), int(image.shape[0] * r)))

        # Resize and pad masks
        resized_masks = []
        for mask in masks:
            resized_mask = cv2.resize(mask.astype(np.uint8), (int(mask.shape[1] * r), int(mask.shape[0] * r)),
                                      interpolation=cv2.INTER_NEAREST)
            if resized_mask.shape[0] < 1024:
                resized_mask = np.pad(resized_mask, ((0, 1024 - resized_mask.shape[0]), (0, 0)), mode="constant")
            if resized_mask.shape[1] < 1024:
                resized_mask = np.pad(resized_mask, ((0, 0), (0, 1024 - resized_mask.shape[1])), mode="constant")
            resized_masks.append(resized_mask)

        if image.shape[0] < 1024:
            image = np.pad(image, ((0, 1024 - image.shape[0]), (0, 0), (0, 0)), mode="constant")
        if image.shape[1] < 1024:
            image = np.pad(image, ((0, 0), (0, 1024 - image.shape[1]), (0, 0)), mode="constant")

        # Return the processed data
        return image, np.array(resized_masks), points, labels

    import pandas as pd

    def train(self, data_path, output_path, **kwargs):
        """
        Train the SAM2 model using the annotations in the train and validation subsets.

        :param data_path: Path to the dataset directory.
        :param output_path: Path to save the trained model and logs.
        """
        num_epochs = self.configuration.get('num_epochs', 100)
        learning_rate = self.configuration.get('learning_rate', 1e-5)
        weight_decay = self.configuration.get('weight_decay', 4e-5)
        # Set intervals and patience
        save_interval = 5  # Save checkpoint every 5 epochs
        patience = 3  # Early stopping patience
        no_improvement_epochs = 0

        # Get train and validation items
        train_items = list(self.model_entity.dataset.items.list(filters=dl.Filters(
            custom_filter=self.model_entity.metadata['system']['subsets']['train']['filter'])).all())

        val_items = list(self.model_entity.dataset.items.list(filters=dl.Filters(
            custom_filter=self.model_entity.metadata['system']['subsets']['validation']['filter'])).all())

        # Set model to train
        self.predictor.model.sam_mask_decoder.train(True)
        self.predictor.model.sam_prompt_encoder.train(True)

        optimizer = torch.optim.AdamW(params=self.predictor.model.parameters(), lr=learning_rate,
                                      weight_decay=weight_decay)
        scaler = torch.cuda.amp.GradScaler()

        best_val_loss = float('inf')  # To track the best validation loss

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            # Initialize epoch losses
            total_epoch_loss = 0.0
            total_seg_loss = 0.0
            total_score_loss = 0.0
            num_batches = 0

            # Training Loop
            self.predictor.model.train()  # Set model to training mode
            for item in train_items:
                # Prepare data batch
                image, masks, points, labels = self.read_batch(item, (item.width, item.height))

                if len(masks) == 0:
                    continue  # Skip items with no annotations

                with torch.cuda.amp.autocast():  # Mixed precision
                    # Set the image in the predictor
                    image_properties = self.predictor.set_image(image=image)

                    # Encode prompts (points and labels)
                    mask_input, unnorm_coords, input_labels, _ = self.predictor._prep_prompts(
                        image_properties=image_properties,
                        point_coords=points, point_labels=labels, box=None, mask_logits=None, normalize_coords=True
                    )
                    sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, input_labels), boxes=None, masks=None
                    )

                    # Decode masks
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                         self.predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                        image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=unnorm_coords.shape[0] > 1,
                        high_res_features=high_res_features,
                    )

                    # Post-process masks
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

                    # Convert ground truth masks to tensor
                    gt_masks = torch.tensor(masks.astype(np.float32)).to(self.device)

                    # Calculate segmentation loss
                    prd_probs = torch.sigmoid(prd_masks[:, 0])  # Probability map
                    seg_loss = (-gt_masks * torch.log(prd_probs + 1e-7) -
                                (1 - gt_masks) * torch.log(1 - prd_probs + 1e-7)).mean()

                    # Calculate score loss (IoU-based)
                    inter = (gt_masks * (prd_probs > 0.5)).sum(1).sum(1)
                    iou = inter / (gt_masks.sum(1).sum(1) + (prd_probs > 0.5).sum(1).sum(1) - inter + 1e-7)
                    score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

                    # Combine losses
                    total_loss = seg_loss + 0.05 * score_loss

                    # Backpropagation and optimization
                    optimizer.zero_grad()
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    # Update epoch losses
                    total_epoch_loss += total_loss.item()
                    total_seg_loss += seg_loss.item()
                    total_score_loss += score_loss.item()
                    num_batches += 1

            # Calculate average training losses for the epoch
            avg_training_loss = total_epoch_loss / num_batches
            avg_seg_loss = total_seg_loss / num_batches
            avg_score_loss = total_score_loss / num_batches

            logger.info(f"Avg training loss: {avg_training_loss}\n"
                  f"Avg segmentation loss: {avg_seg_loss}\n"
                  f"Avg score loss: {avg_score_loss}")

            # Validation Loop
            logger.info("Validating...")
            self.predictor.model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # Disable gradient calculation for validation
                for item in val_items:
                    image, masks, points, labels = self.read_batch(item, (item.width, item.height))

                    if len(masks) == 0:
                        continue

                    # Set the image in the predictor
                    image_properties = self.predictor.set_image(image=image)

                    # Encode prompts
                    mask_input, unnorm_coords, input_labels, _ = self.predictor._prep_prompts(
                        image_properties=image_properties,
                        point_coords=points, point_labels=labels, box=None, mask_logits=None, normalize_coords=True
                    )
                    sparse_embeddings, dense_embeddings = self.predictor.model.sam_prompt_encoder(
                        points=(unnorm_coords, input_labels), boxes=None, masks=None
                    )

                    # Decode masks
                    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in
                                         self.predictor._features["high_res_feats"]]
                    low_res_masks, prd_scores, _, _ = self.predictor.model.sam_mask_decoder(
                        image_embeddings=self.predictor._features["image_embed"][-1].unsqueeze(0),
                        image_pe=self.predictor.model.sam_prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                        repeat_image=unnorm_coords.shape[0] > 1,
                        high_res_features=high_res_features,
                    )

                    # Post-process masks
                    prd_masks = self.predictor._transforms.postprocess_masks(low_res_masks, self.predictor._orig_hw[-1])

                    # Convert ground truth masks to tensor
                    gt_masks = torch.tensor(masks.astype(np.float32)).to(self.device)

                    # Calculate segmentation loss
                    prd_probs = torch.sigmoid(prd_masks[:, 0])  # Probability map
                    seg_loss = (-gt_masks * torch.log(prd_probs + 1e-7) -
                                (1 - gt_masks) * torch.log(1 - prd_probs + 1e-7)).mean()

                    # Combine losses
                    val_loss += seg_loss.item()

            val_loss /= len(val_items)
            logger.info(f"Validation Loss: {val_loss:.4f}")

            # Save regular checkpoints
            if (epoch + 1) % save_interval == 0:
                torch.save(self.predictor.model.state_dict(), f"{output_path}/sam2_model_epoch_{epoch + 1}.torch")
                logger.info(f"Checkpoint saved at epoch {epoch + 1}.")

            # Early stopping
            if val_loss < best_val_loss:
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
