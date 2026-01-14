import json
import os
import random
from pathlib import Path
from typing import Any, List, Tuple, Union
import torchvision

from IPython.display import Video
import torch
import cv2
import gdown
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from IPython.display import YouTubeVideo
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

import super_gradients
from super_gradients.training import models, Trainer
from super_gradients.common.object_names import Models

from super_gradients.common.decorators.factory_decorator import resolve_param
from super_gradients.common.factories.target_generator_factory import TargetGeneratorsFactory
from super_gradients.common.factories.transforms_factory import TransformsFactory
from super_gradients.common.object_names import Datasets
from super_gradients.common.registry import register_dataset
from super_gradients.training.transforms.keypoint_transforms import AbstractKeypointTransform
from super_gradients.training.samples import PoseEstimationSample

from super_gradients.training.datasets.pose_estimation_datasets.abstract_pose_estimation_dataset import AbstractPoseEstimationDataset

from super_gradients.training.datasets.pose_estimation_datasets import YoloNASPoseCollateFN

from super_gradients.training.transforms.keypoints import (
    KeypointsHSV,
    KeypointsBrightnessContrast,
    KeypointsMosaic,
    KeypointsRandomAffineTransform,
    KeypointsLongestMaxSize,
    KeypointsPadIfNeeded,
    KeypointsImageStandardize,
    KeypointsImageNormalize,
    KeypointsRemoveSmallObjects
)


from super_gradients.training.models.pose_estimation_models.yolo_nas_pose import YoloNASPosePostPredictionCallback
from super_gradients.training.utils.callbacks import ExtremeBatchPoseEstimationVisualizationCallback, Phase
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.metrics import PoseEstimationMetrics

import omegaconf as oc

from pytube import YouTube

class PoseEstimationDataset(AbstractPoseEstimationDataset):
    """
    Dataset class for training pose estimation models on Animal Pose dataset.
    """

    @resolve_param("transforms", TransformsFactory())
    def __init__(
        self,
        data_dir: str,
        images_dir: str,
        json_file: str,
        transforms: List[AbstractKeypointTransform],
        edge_links: Union[List[Tuple[int, int]], np.ndarray],
        edge_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
        keypoint_colors: Union[List[Tuple[int, int, int]], np.ndarray, None],
    ):
        split_json_file = os.path.join(data_dir, json_file)

        with open(split_json_file, "r") as f:
            json_annotations = json.load(f)

        # Assuming first category has keypoints
        joints = json_annotations["categories"][1]["keypoints"]
        num_joints = len(joints)

        super().__init__(
            transforms=transforms,
            num_joints=num_joints,
            edge_links=edge_links,
            edge_colors=edge_colors,
            keypoint_colors=keypoint_colors,
        )

        self.num_joints = num_joints
        print(f"Number of joints: {self.num_joints}")

        self.image_ids = []
        self.image_files = []
        self.annotations = []

        # Preprocess: make a mapping from image_id -> list of annotations
        from collections import defaultdict

        image_id_to_annotations = defaultdict(list)
        for ann in json_annotations["annotations"]:
            image_id_to_annotations[ann["image_id"]].append(ann)

        for image in json_annotations["images"]:
            image_id = image["id"]
            file_path = os.path.join(data_dir, images_dir, image["file_name"])

            image_annotations = image_id_to_annotations.get(image_id, [])

            keypoints_per_image = []
            bboxes_per_image = []

            for ann in image_annotations:
                if "keypoints" not in ann:
                    continue
                keypoints = np.array(ann["keypoints"]).reshape(self.num_joints, 3)
                x1, y1, w, h = ann["bbox"]
                bbox_xywh = np.array([x1, y1, w, h])

                keypoints_per_image.append(keypoints)
                bboxes_per_image.append(bbox_xywh)

            if keypoints_per_image:
                self.image_ids.append(image_id)
                self.image_files.append(file_path)

                keypoints_per_image = np.array(keypoints_per_image, dtype=np.float32).reshape(-1, self.num_joints, 3)
                bboxes_per_image = np.array(bboxes_per_image, dtype=np.float32).reshape(-1, 4)

                annotation = (keypoints_per_image, bboxes_per_image)
                self.annotations.append(annotation)
    def __len__(self):
        return len(self.annotations)  # <<<< MATCHES THE CORRECT NUMBER!!

    def load_sample(self, index) -> PoseEstimationSample:
        file_path = self.image_files[index]
        
        gt_joints, gt_bboxes = self.annotations[index]  # boxes in xywh format
        
        gt_areas = np.array([box[2] * box[3] for box in gt_bboxes], dtype=np.float32)
        gt_iscrowd = np.array([0] * len(gt_joints), dtype=bool)

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        mask = np.ones(image.shape[:2], dtype=np.float32)

        return PoseEstimationSample(
            image=image, mask=mask, joints=gt_joints, areas=gt_areas, bboxes_xywh=gt_bboxes, is_crowd=gt_iscrowd, additional_samples=None
        )

def open_file(file_path: str) -> Union[dict, list, None]:

    try:
        with open(file_path, 'r') as file:
            if file_path.endswith('.json'):
                print('success')
                return json.load(file)
            elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                print('success')
                return yaml.safe_load(file)
            else:
                raise ValueError(f'Unsupported file format: {file_path}')
    except Exception as e:
        print(f'An Error Occurred: {e}')
        return None
    
def plot_random_images(data, image_base_dir="F:\Desktop\proj"):
    """
    Plots 5 random images for each category from the provided dataset.

    Parameters:
    - data: The JSON dataset containing image, annotation, and category details.
    - image_base_dir: The base directory where the images are located.
    """

    # Create a dictionary to map image IDs to filenames
    image_id_to_filename = {image['id']: image['file_name'] for image in data['images']}

    # Extracting image_ids for each category
    category_image_ids = {}
    for category in data['categories']:
        category_id = category['id']
        category_name = category['name']
        category_image_ids[category_name] = [anno['image_id'] for anno in data['annotations'] if anno['category_id'] == category_id]

    # Randomly select 5 image_ids for each category
    random_selected_ids = {}
    for category_name, ids in category_image_ids.items():
        random_selected_ids[category_name] = random.sample(ids, min(5, len(ids)))

    # Number of categories
    num_categories = len(random_selected_ids)

    # Create a figure to plot the images
    fig, axes = plt.subplots(num_categories, 5, figsize=(20, num_categories * 3))
    if num_categories == 1:  # If there is only one category, axes will be 1D
        axes = [axes]

    for i, (category_name, ids) in enumerate(random_selected_ids.items()):
        for j, image_id in enumerate(ids):
            # Get the filename using the image_id_to_filename dictionary
            filename = image_id_to_filename.get(image_id, "Image_Not_Found.jpg")

            # Load and plot the image
            img_path = os.path.join(image_base_dir, filename)
            print(f"Loading image: {img_path}")  # Debugging line to check file paths
            try:
                img = mpimg.imread(img_path)
                axes[i][j].imshow(img)
            except FileNotFoundError:
                print(f"Image not found: {img_path}")  # Debugging line to show when image is not found
                axes[i][j].imshow(np.zeros((100, 100, 3)))  # Show an empty image if file is not found
            axes[i][j].axis('off')
            if j == 0:
                axes[i][j].set_title(category_name)

    plt.tight_layout()
    plt.show()

def train():
    trainer.train(model=yolo_nas_pose,
                training_params=train_params,
                train_loader=train_dataloader,
                valid_loader=val_dataloader
                )

def videoConverter(path):
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter('output_predicted_video.mp4', fourcc, 30.0, (640, 640))

    frames = []
    batch_size = 16
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 640))
        frames.append(frame)

        if len(frames) == batch_size:
            predictions = best_model.predict(frames, conf=0.5)

            for pred in predictions:
                out.write(pred.draw())  # or pred.show() if you want to view live

            frames = []  # reset
     # Handle any remaining frames after the last full batch
    if frames:
        predictions = best_model.predict(frames, conf=0.5)
        for pred in predictions:
            result_img = pred.draw()
            out.write(result_img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    

    print(torch.__version__)
    print(torchvision.__version__)
    print(torch.cuda.is_available())   # should print True
    print(torch.cuda.get_device_name(0))  # should print your GPU name

    CHECKPOINT_DIR = r'F:\Desktop\proj\results'
    trainer = Trainer(experiment_name='first_yn_pose_run', ckpt_root_dir=CHECKPOINT_DIR)


    annotations = open_file('')
    config = open_file('F:\Desktop\proj\coco_pose_common_dataset_params.yaml')

    # plot_random_images(data=annotations, image_base_dir="")

    train_annotations = open_file(r'F:\Desktop\proj\dataset\train\_annotations.coco.json')
    val_annotations = open_file(r'F:\Desktop\proj\dataset\valid\_annotations.coco.json')
    test_annotations = open_file(r'F:\Desktop\proj\dataset\test\_annotations.coco.json')
    print(len(train_annotations['images']))
    print(len(train_annotations['annotations']))
    print(len(val_annotations['annotations']))
    print(len(test_annotations['annotations']))

    keypoints_hsv = KeypointsHSV(prob=0.5, hgain=20, sgain=20, vgain=20)

    keypoints_brightness_contrast = KeypointsBrightnessContrast(prob=0.5,
                                                                brightness_range=[0.8, 1.2],
                                                                contrast_range=[0.8, 1.2]
                                                                )

    keypoints_mosaic = KeypointsMosaic(prob=0.8)

    keypoints_random_affine_transform = KeypointsRandomAffineTransform(max_rotation=0,
                                                                    min_scale=0.5,
                                                                    max_scale=1.5,
                                                                    max_translate=0.1,
                                                                    image_pad_value=127,
                                                                    mask_pad_value=1,
                                                                    prob=0.75,
                                                                    interpolation_mode=[0, 1, 2, 3, 4]
                                                                    )

    keypoints_longest_max_size = KeypointsLongestMaxSize(max_height=640, max_width=640)

    keypoints_pad_if_needed = KeypointsPadIfNeeded(min_height=640,
                                                min_width=640,
                                                image_pad_value=[127, 127, 127],
                                                mask_pad_value=1,
                                                padding_mode='bottom_right'
                                                )

    keypoints_image_standardize = KeypointsImageStandardize(max_value=255)

    # keypoints_image_normalize = KeypointsImageNormalize(mean=[0.485, 0.456, 0.406],
    #                                                     std=[0.229, 0.224, 0.225]
    #                                                     )

    keypoints_remove_small_objects = KeypointsRemoveSmallObjects(min_instance_area=1,
                                                                min_visible_keypoints=1
                                                                )
    
    train_transforms = [
        keypoints_hsv,
        keypoints_brightness_contrast,
        keypoints_mosaic,
        keypoints_random_affine_transform,
        keypoints_longest_max_size,
        keypoints_pad_if_needed,
        keypoints_image_standardize,
        keypoints_remove_small_objects
    ]

    val_transforms = [
        keypoints_longest_max_size,
        keypoints_pad_if_needed,
        keypoints_image_standardize,
    ]

    data_path = r"F:\Desktop\proj\dataset"

    # Create instances of the dataset
    train_dataset = PoseEstimationDataset(
        data_dir=data_path,
        images_dir= data_path + '/train',
        json_file= data_path + '/train/_annotations.coco.json',
        transforms=train_transforms,
        edge_links = config['skeleton'],
        edge_colors = config['edge_colors'],
        keypoint_colors = config['keypoint_colors']
        )

    val_dataset = PoseEstimationDataset(
        data_dir=data_path,
        images_dir= data_path + '/valid',
        json_file= data_path + '/valid/_annotations.coco.json',
        transforms=val_transforms,
        edge_links = config['skeleton'],
        edge_colors = config['edge_colors'],
        keypoint_colors = config['keypoint_colors']
        )

    test_dataset = PoseEstimationDataset(
        data_dir=data_path,
        images_dir= data_path + '/test',
        json_file= data_path + '/test/_annotations.coco.json',
        transforms=val_transforms,
        edge_links = config['skeleton'],
        edge_colors = config['edge_colors'],
        keypoint_colors = config['keypoint_colors']
        )
    
    #train_dataset.image_ids = train_dataset.image_ids[:50]
    #train_dataset.image_files = train_dataset.image_files[:50]
    #train_dataset.annotations = train_dataset.annotations[:50]

    #val_dataset.image_ids = train_dataset.image_ids[:50]
    #val_dataset.image_files = train_dataset.image_files[:50]
    #val_dataset.annotations = train_dataset.annotations[:50]
       
    #test_dataset.image_ids = train_dataset.image_ids[:50]
    #test_dataset.image_files = train_dataset.image_files[:50]
    #test_dataset.annotations = train_dataset.annotations[:50]
       
        


    # Create dataloaders
    train_dataloader_params = {
        'shuffle': True,
        'batch_size':8,
        'drop_last': True,
        'pin_memory': False,
        'collate_fn': YoloNASPoseCollateFN()
        }

    val_dataloader_params = {
        'shuffle': True,
        'batch_size': 8,
        'drop_last': True,
        'pin_memory': False,
        'collate_fn': YoloNASPoseCollateFN()
        }
    
    train_dataloader = DataLoader(train_dataset, **train_dataloader_params)

    val_dataloader = DataLoader(val_dataset, **val_dataloader_params)

    test_dataloader = DataLoader(test_dataset, **val_dataloader_params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo_nas_pose = models.get("yolo_nas_pose_l", num_classes= config['num_joints'], pretrained_weights= "coco_pose").cuda()

    post_prediction_callback = YoloNASPosePostPredictionCallback(
        pose_confidence_threshold = 0.01,
        nms_iou_threshold = 0.7,
        pre_nms_max_predictions = 300,
        post_nms_max_predictions = 30,
        )

    metrics = PoseEstimationMetrics(
        num_joints = config['num_joints'],
        oks_sigmas = config['oks_sigmas'],
        max_objects_per_image = 30,
        post_prediction_callback = post_prediction_callback,
        )

    visualization_callback = ExtremeBatchPoseEstimationVisualizationCallback(
        keypoint_colors = config["keypoint_colors"],
        edge_colors = config['edge_colors'],
        edge_links = config['skeleton'],
        loss_to_monitor = "YoloNASPoseLoss/loss",
        max = True,
        freq = 1,
        max_images = 1,
        enable_on_train_loader = True,
        enable_on_valid_loader = True,
        post_prediction_callback = post_prediction_callback,
        )

    early_stop = EarlyStop(
        phase = Phase.VALIDATION_EPOCH_END,
        monitor = "AP",
        mode = "max",
        min_delta = 0.0001,
        patience = 100,
        verbose = True,
        )

    train_params = {
        "warmup_mode": "LinearBatchLRWarmup",
        "warmup_initial_lr": 1e-8,
        "lr_warmup_epochs": 1,
        "initial_lr": 5e-5,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 5e-3,
        "resume": True,
        "max_epochs": 10,
        "zero_weight_decay_on_bias_and_bn": True,
        "batch_accumulate": 1,
        "average_best_models": True,
        "save_ckpt_epoch_list": [5, 10, 15, 20],
        "loss": "yolo_nas_pose_loss",
        "criterion_params": {
            "oks_sigmas": config['oks_sigmas'],
            "classification_loss_weight": 1.0,
            "classification_loss_type": "focal",
            "regression_iou_loss_type": "ciou",
            "iou_loss_weight": 2.5,
            "dfl_loss_weight": 0.01,
            "pose_cls_loss_weight": 1.0,
            "pose_reg_loss_weight": 34.0,
            "pose_classification_loss_type": "focal",
            "rescale_pose_loss_with_assigned_score": True,
            "assigner_multiply_by_pose_oks": True,
        },
        "optimizer": "AdamW",
        "optimizer_params": {
            "weight_decay": 0.000001
        },
        "ema": True,
        "ema_params": {
            "decay": 0.997,
            "decay_type": "threshold"
        },
        "mixed_precision": True,
        "sync_bn": False,
        "valid_metrics_list": [metrics],
        "phase_callbacks": [visualization_callback, early_stop],
        "pre_prediction_callback": None,
        "metric_to_watch": "AP",
        "greater_metric_to_watch_is_better": True,
        "_convert_": "all"
    }


    train_params['resume'] = True

    train()

    #best_model = models.get('yolo_nas_pose_l', num_classes=config['num_joints'], checkpoint_path=r"F:\Desktop\proj\results\first_yn_pose_run\RUN_20250428_225442_723615\ckpt_best.pth")
    
    

    post_prediction_callback = YoloNASPosePostPredictionCallback(
        pose_confidence_threshold = 0.01,
        nms_iou_threshold = 0.7,
        pre_nms_max_predictions = 300,
        post_nms_max_predictions = 30,
    )

    metrics = PoseEstimationMetrics(
        num_joints = config['num_joints'],
        oks_sigmas = config['oks_sigmas'],
        max_objects_per_image = 30,
        post_prediction_callback = post_prediction_callback,
    )

    
    #print(trainer.test(model=best_model,
    #           test_loader=test_dataloader,
    #         test_metrics_list=metrics))
    
    
    img_url = r"F:\Desktop\proj\dataset\valid\812_jpg.rf.e1c166e364a3ab6e81d83c72a656574a.jpg"
    #best_model.predict(img_url, conf = 0.30).show()

    
    #videoConverter(r"F:\Desktop\proj\video_test\test.mp4")
    #print("this is the video:")
    #Video("F:\Desktop\proj\output_predicted_video.mp4", embed=True)