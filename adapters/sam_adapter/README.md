# SAM2 Model Adapter

This is a Dataloop adapter for the Segment Anything Model 2 (SAM2), enabling both inference and fine-tuning capabilities.

## Prediction

The model supports three prediction modes:

### 1. Zero-shot Segmentation (No Annotations)
- When no annotations are provided, a bounding box of the entire image is passed to the model
- Returns a binary mask for the most prominent object

### 2. Box-guided Segmentation
- Users can provide bounding boxes around areas of interest
- The model generates segmentation masks within the specified boxes

### 3. Point-based Segmentation
- Users can input points, with each point indicating a specific location inside a different object to be segmented
- Returns a segmentation mask for each object containing an indicated point

### 4. Multi-points Segmentation
- Users can input multiple points, each labeled as "inside" or "outside" to indicate their position relative to objects.
- The model uses these labeled points to generate segmentation masks, refining the segmentation based on the provided labels.
- **Configuration Requirement**: Ensure that the `multi_points_prediction` setting is set to `true` in the model configuration to use this mode.
- **Output Type**: The `output_type` setting can be set to either `segment` or `polygon` in the model configuration, depending on whether you want segmentation masks or polygon annotations as output.

## Training

The adapter supports fine-tuning of the SAM2 model using two types of input annotations:

### Input Annotation Types
1. **Images with Masks**
   - Direct training using predefined binary masks
   - Optimal for datasets with existing segmentation masks

2. **Images with Polygons**
   - Automatic conversion of polygon annotations to binary masks
   - Enables training with polygon-based datasets

### Configuration Options
- `model_cfg`: Model configuration file (choose 'sam2_hiera_s.yaml' for small model or 'sam2_hiera_l.yaml' for large model, default: 'sam2_hiera_s.yaml')
- `save_model_name`: Name of the model to save (default: 'best_sam2_model.torch')
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Learning rate for optimization (default: 1e-5) 
- `weight_decay`: Weight decay for regularization (default: 4e-5)
- `batch_size`: Batch size for training (default: 4)
- `save_interval`: Number of epochs between model checkpoints (default: 10)
- `patience`: Number of epochs to wait for improvement before early stopping (default: 10)
- `multi_points_prediction`: Whether to use multi-points prediction (default: false)
- `output_type`: Whether to use segmentation masks or polygon annotations as output (default: 'segment')

### Training Features
- Mixed precision training for improved performance
- Early stopping with configurable patience
- Regular model checkpointing
- Loss function combining segmentation and IoU-based score losses
- Fine-tunes only the mask decoder and prompt encoder while keeping the image encoder frozen

### Data Processing
- Automatic handling of image resizing and padding
- Generation of point prompts from segmentation masks for training
- Conversion of various annotation formats to model-compatible inputs

The trained model can be saved and loaded for future use within the Dataloop platform.