import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import datetime, timedelta
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import argparse
import math
import boto3
import requests
import signal
import sys
from botocore.exceptions import ClientError
import shutil
import tarfile
import io
from PIL import Image


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Training configuration
class Config:
    num_epochs = 150
    batch_size = 450
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4
    num_workers = 16
    subset_size = None
    print_freq = 100

    # Add gradient accumulation steps if needed
    accum_iter = 1  # Can be increased if memory allows

    # Add mixed precision training parameters
    use_amp = True  # Enable automatic mixed precision


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class S3ImageNetDataset(Dataset):
    def __init__(self, bucket_name, prefix, transform=None):
        self.s3_client = boto3.client('s3')
        self.bucket = bucket_name
        self.prefix = prefix
        self.transform = transform
        
        # List all objects in the bucket with the given prefix
        self.image_keys = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        print(f"Loading image list from s3://{bucket_name}/{prefix}")
        for page in tqdm(paginator.paginate(Bucket=bucket_name, Prefix=prefix)):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.JPEG'):
                        self.image_keys.append(obj['Key'])
        
        # Extract class labels from paths
        self.class_to_idx = {}
        class_set = set()
        for key in self.image_keys:
            class_name = key.split('/')[-2]  # Assuming format: prefix/class_name/image.JPEG
            class_set.add(class_name)
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
        
        # Create sorted list of classes
        self.classes = sorted(list(class_set))
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        print(f"Found {len(self.image_keys)} images in {len(self.classes)} classes")

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        # Get image from S3
        key = self.image_keys[idx]
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Get class label
            class_name = key.split('/')[-2]
            label = self.class_to_idx[class_name]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
            
        except Exception as e:
            print(f"Error loading image {key}: {str(e)}")
            # Return a default image or skip
            return self.__getitem__((idx + 1) % len(self))


def get_data_loaders(subset_size=None, distributed=False, world_size=None, rank=None):
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Modified data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.5),
    ])

    # Modified transform for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Create S3 datasets
    train_dataset = S3ImageNetDataset(
        bucket_name="imagenet-dataset-nikhil",
        prefix="imagenet/ILSVRC/Data/CLS-LOC/train",
        transform=train_transform
    )
    
    val_dataset = S3ImageNetDataset(
        bucket_name="imagenet-dataset-nikhil",
        prefix="imagenet/ILSVRC/Data/CLS-LOC/val",
        transform=val_transform
    )

    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        val_indices = torch.randperm(len(val_dataset))[:subset_size // 10]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_sampler = None
    val_sampler = None

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=Config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
        prefetch_factor=2
    )

    return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    epoch_start_time = datetime.now()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create GradScaler for mixed precision training
    scaler = torch.amp.GradScaler('cuda') if Config.use_amp else None

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for i, data in enumerate(pbar):
        try:
            images, targets = data
            images, targets = images.to(device), targets.to(device)

            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=Config.use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = (
                    loss / Config.accum_iter
                )  # Normalize loss for gradient accumulation

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if ((i + 1) % Config.accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * Config.accum_iter
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if i % Config.print_freq == 0:
                accuracy = 100.0 * correct / total
                pbar.set_postfix(
                    {
                        "loss": running_loss / (i + 1),
                        "acc": f"{accuracy:.2f}%",
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
        except Exception as e:
            print(f"Error in batch {i}: {str(e)}")
            torch.cuda.empty_cache()  # Clear cache after error
            continue

    # Calculate epoch time and return metrics
    epoch_time = datetime.now() - epoch_start_time
    epoch_metrics = {
        "time": epoch_time,
        "loss": running_loss / len(train_loader),
        "accuracy": 100.0 * correct / total,
    }
    return epoch_metrics


def validate(model, val_loader, criterion, device):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=Config.use_amp):
        for images, targets in tqdm(val_loader, desc="Validating"):
            images, targets = images.to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)

            # Compute top-1 and top-5 accuracy
            maxk = max((1, 5))
            batch_size = targets.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))

            # Top-1 accuracy
            top1_acc = correct[0].float().sum() * 100.0 / batch_size
            top1.update(top1_acc.item(), batch_size)

            # Top-5 accuracy
            top5_acc = correct[:5].float().sum() * 100.0 / batch_size
            top5.update(top5_acc.item(), batch_size)

            losses.update(loss.item(), batch_size)

    return top1.avg, top5.avg, losses.avg


# Add ResNet building blocks
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# Replace the model creation in main() with this:
def create_resnet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


# Add logging setup function
def setup_logging(log_dir, aws_handler):
    # Create local log directory
    os.makedirs(log_dir, exist_ok=True)

    # Setup formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # File handler for local logs
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)

    # Stream handler for console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    # S3 handler for remote logs
    s3_handler = S3LogHandler(
        aws_handler.s3_client,
        aws_handler.bucket_name,
        f"{aws_handler.model_prefix}/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    s3_handler.setFormatter(formatter)

    # Add all handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addHandler(s3_handler)

    return logger


# Add distributed training setup
def setup_distributed():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1)
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "-1"

    args.local_rank = int(os.environ["LOCAL_RANK"])

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.world_size = args.nodes

    return args


# Add this function to get dataset statistics
def get_dataset_stats(train_loader, val_loader):
    return {
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "num_classes": (
            len(train_loader.dataset.classes)
            if hasattr(train_loader.dataset, "classes")
            else len(train_loader.dataset.dataset.classes)
        ),
        "batch_size": train_loader.batch_size,
        "num_train_batches": len(train_loader),  # This is total_samples/batch_size rounded up
        "num_val_batches": len(val_loader),
    }


class AWSSpotHandler:
    def __init__(self, bucket_name, model_prefix, sns_topic_arn=None):
        self.s3_client = boto3.client("s3")
        self.sns_client = boto3.client("sns")
        self.bucket_name = bucket_name
        self.model_prefix = model_prefix
        self.sns_topic_arn = sns_topic_arn
        self.instance_id = self._get_instance_id()

    def _get_instance_id(self):
        try:
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/instance-id", timeout=0.1
            )
            return response.text
        except:
            return None

    def check_spot_termination(self):
        try:
            response = requests.get(
                "http://169.254.169.254/latest/meta-data/spot/termination-time",
                timeout=0.1,
            )
            return response.status_code == 200
        except:
            return False

    def save_checkpoint(self, state, is_best=False, filename="checkpoint.pth.tar"):
        """Save checkpoint to S3"""
        local_path = os.path.join("checkpoints", filename)
        os.makedirs("checkpoints", exist_ok=True)

        # Save locally first
        torch.save(state, local_path)

        # Upload to S3
        s3_path = f"{self.model_prefix}/checkpoints/{filename}"
        try:
            self.s3_client.upload_file(local_path, self.bucket_name, s3_path)
            if is_best:
                best_path = f"{self.model_prefix}/checkpoints/model_best.pth.tar"
                self.s3_client.upload_file(local_path, self.bucket_name, best_path)
        except Exception as e:
            print(f"Failed to upload checkpoint to S3: {e}")

    def load_checkpoint(self):
        """Load latest checkpoint from S3"""
        try:
            # List all checkpoints
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=f"{self.model_prefix}/checkpoints/"
            )

            if "Contents" not in response:
                print("No checkpoints found in S3")
                return None

            # Get the latest checkpoint
            checkpoints = sorted(
                response["Contents"],
                key=lambda x: x["LastModified"],
                reverse=True
            )

            if not checkpoints:
                print("No checkpoints found in sorted list")
                return None

            # Download the latest checkpoint
            latest = checkpoints[0]["Key"]
            # Add rank to filename to avoid conflicts
            rank = int(os.environ.get('LOCAL_RANK', 0))
            local_path = os.path.join("checkpoints", f"rank{rank}_{os.path.basename(latest)}")
            
            # Ensure directory exists
            os.makedirs("checkpoints", exist_ok=True)
            
            print(f"Downloading checkpoint from s3://{self.bucket_name}/{latest}")
            
            # Download directly to final path without temporary file
            self.s3_client.download_file(
                self.bucket_name,
                latest,
                local_path
            )

            if not os.path.exists(local_path):
                print(f"Error: Downloaded file not found at {local_path}")
                return None

            print(f"Loading checkpoint from {local_path}")
            checkpoint = torch.load(local_path)
            print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
            
            # Clean up after loading
            try:
                os.remove(local_path)
            except Exception as e:
                print(f"Error removing temporary checkpoint file: {e}")
            
            return checkpoint

        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def send_notification(self, message, subject="Training Status"):
        """Send SNS notification"""
        if self.sns_topic_arn:
            try:
                self.sns_client.publish(
                    TopicArn=self.sns_topic_arn, Message=message, Subject=subject
                )
            except Exception as e:
                print(f"Failed to send notification: {e}")

    def cleanup(self):
        """Clean up local checkpoints"""
        try:
            if os.path.exists("checkpoints"):
                shutil.rmtree("checkpoints")
        except Exception as e:
            print(f"Error cleaning up checkpoints: {e}")

    def save_tensorboard_logs(self, log_dir):
        """Save TensorBoard logs to S3"""
        try:
            # Create a tarfile of the logs
            tar_path = f"{log_dir}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(log_dir)

            # Upload to S3
            s3_path = (
                f"{self.model_prefix}/tensorboard_logs/{os.path.basename(tar_path)}"
            )
            self.s3_client.upload_file(tar_path, self.bucket_name, s3_path)

            # Clean up local tar file
            os.remove(tar_path)
        except Exception as e:
            print(f"Failed to save TensorBoard logs to S3: {e}")


# Add this class after the AWSSpotHandler class
class S3LogHandler(logging.Handler):
    def __init__(self, s3_client, bucket_name, log_path):
        super().__init__()
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        self.log_path = log_path
        self.buffer = []

    def emit(self, record):
        try:
            msg = self.format(record)
            self.buffer.append(msg + "\n")

            # Upload to S3 every 10 log messages or if it's an error/warning
            if len(self.buffer) >= 10 or record.levelno >= logging.WARNING:
                self.flush()
        except Exception:
            self.handleError(record)

    def flush(self):
        if not self.buffer:
            return

        try:
            # First, try to download existing logs
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name, Key=self.log_path
                )
                existing_logs = response["Body"].read().decode("utf-8")
            except self.s3_client.exceptions.NoSuchKey:
                existing_logs = ""

            # Append new logs
            all_logs = existing_logs + "".join(self.buffer)

            # Upload combined logs back to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=self.log_path,
                Body=all_logs.encode("utf-8"),
            )
            self.buffer = []
        except Exception as e:
            print(f"Error writing logs to S3: {e}")


# Modify the main function
def main():
    # AWS configuration
    aws_config = {
        "bucket_name": "imagenet-dataset-nikhil",
        "model_prefix": "resnet50-training",
        "sns_topic_arn": "arn:aws:sns:us-east-2:515966496154:imagenet_s3",  # Optional
    }

    # Initialize AWS handler
    aws_handler = AWSSpotHandler(**aws_config)

    start_time = datetime.now()
    args = setup_distributed()

    try:
        # Setup distributed training
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/resnet50_{timestamp}"
        logger = None

        if args.local_rank <= 0:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir)
            logger = setup_logging(log_dir, aws_handler)
            logger.info(f"Starting training on {socket.gethostname()}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
            logger.info(f"Training started at: {start_time}")

        # Ensure all ranks have access to a logger
        if args.local_rank > 0:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.INFO)
            logger.addHandler(logging.StreamHandler())

        set_seed()
        # Get data loaders
        train_loader, val_loader, train_sampler = get_data_loaders(
            subset_size=Config.subset_size,
            distributed=(args.local_rank != -1),
            world_size=args.world_size,
            rank=args.local_rank,
        )

        # Log dataset statistics
        if args.local_rank <= 0:
            dataset_stats = get_dataset_stats(train_loader, val_loader)
            logger.info("Dataset Statistics:")
            logger.info(f"Training samples: {dataset_stats['num_train_samples']}")
            logger.info(f"Validation samples: {dataset_stats['num_val_samples']}")
            logger.info(f"Number of classes: {dataset_stats['num_classes']}")
            logger.info(f"Batch size: {dataset_stats['batch_size']}")
            logger.info(
                f"Training batches per epoch: {dataset_stats['num_train_batches']}"
            )
            logger.info(
                f"Validation batches per epoch: {dataset_stats['num_val_batches']}"
            )

            # Save dataset stats to S3
            try:
                stats_path = f"{aws_handler.model_prefix}/dataset_stats.json"
                aws_handler.s3_client.put_object(
                    Bucket=aws_handler.bucket_name,
                    Key=stats_path,
                    Body=json.dumps(dataset_stats, indent=2),
                )
            except Exception as e:
                logger.warning(f"Failed to save dataset stats to S3: {e}")

        # Create model
        model = create_resnet50()
        if args.local_rank != -1:
            model = DDP(model.to(device), device_ids=[args.local_rank])
        else:
            model = torch.nn.DataParallel(model).to(device)

        # Initialize training components
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.SGD(
            model.parameters(),
            lr=Config.learning_rate,
            momentum=Config.momentum,
            weight_decay=Config.weight_decay,
            nesterov=True,
        )

        # Cosine annealing with warmup
        warmup_epochs = 5

        def warmup_lr_scheduler(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            return 0.5 * (
                1
                + math.cos(
                    math.pi
                    * (epoch - warmup_epochs)
                    / (Config.num_epochs - warmup_epochs)
                )
            )

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=Config.num_epochs
        # )

        # Try to load checkpoint
        start_epoch = 0
        best_acc = 0
        # Only rank 0 loads checkpoint first
        if args.local_rank == 0:
            checkpoint = aws_handler.load_checkpoint()
            # Broadcast success/failure to other ranks
            success = torch.tensor([checkpoint is not None], device=device)
        else:
            success = torch.tensor([False], device=device)
            checkpoint = None
            
        # Broadcast success to all ranks
        if args.local_rank != -1:
            dist.broadcast(success, 0)
            
        # If rank 0 succeeded, other ranks should also load
        if success.item():
            if args.local_rank > 0:
                checkpoint = aws_handler.load_checkpoint()

        if checkpoint:
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                best_acc = checkpoint.get("best_acc", 0)
                if args.local_rank <= 0:
                    logger.info(f"Successfully resumed from epoch {start_epoch}")
                    logger.info(f"Previous best accuracy: {best_acc:.2f}%")
            except Exception as e:
                logger.warning(f"Error loading checkpoint state: {str(e)}")
                logger.warning("Starting training from scratch")
                start_epoch = 0
                best_acc = 0
        else:
            logger.info("No checkpoint found, starting training from scratch")

        # Training loop
        total_training_time = timedelta()
        for epoch in range(start_epoch, Config.num_epochs):
            if args.local_rank <= 0:
                logger.info(f"Starting epoch {epoch}")

            # Set train sampler epoch for distributed training
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Check for spot termination
            if aws_handler.check_spot_termination():
                message = f"Spot instance {aws_handler.instance_id} is being terminated. Saving checkpoint..."
                if args.local_rank <= 0:
                    logger.warning(message)
                    aws_handler.send_notification(message, "Spot Instance Termination")

                # Save emergency checkpoint
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                if args.local_rank <= 0:
                    aws_handler.save_checkpoint(
                        state,
                        filename=f'emergency_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth.tar',
                    )
                sys.exit(0)

            # Train for one epoch and get metrics
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, epoch, device
            )
            total_training_time += train_metrics["time"]

            if args.local_rank <= 0:
                # Log training metrics
                logger.info(
                    f"Epoch {epoch} completed in {train_metrics['time']}, "
                    f"Training Loss: {train_metrics['loss']:.4f}, "
                    f"Training Accuracy: {train_metrics['accuracy']:.2f}%"
                )

                # Get all three metrics from validate
                top1_acc, top5_acc, val_loss = validate(
                    model, val_loader, criterion, device
                )

                # Log validation metrics
                logger.info(
                    f"Validation metrics - "
                    f"Top1 Acc: {top1_acc:.2f}%, "
                    f"Top5 Acc: {top5_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}"
                )

                # Log cumulative time
                logger.info(f"Total training time so far: {total_training_time}")

                # Log to tensorboard
                writer.add_scalar("Training/Loss", train_metrics["loss"], epoch)
                writer.add_scalar("Training/Accuracy", train_metrics["accuracy"], epoch)
                writer.add_scalar(
                    "Training/Time", train_metrics["time"].total_seconds(), epoch
                )
                writer.add_scalar("Validation/Top1_Accuracy", top1_acc, epoch)
                writer.add_scalar("Validation/Top5_Accuracy", top5_acc, epoch)
                writer.add_scalar("Validation/Loss", val_loss, epoch)

                is_best = top1_acc > best_acc
                best_acc = max(top1_acc, best_acc)

                # Save checkpoint
                state = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "top1_accuracy": top1_acc,
                    "top5_accuracy": top5_acc,
                }
                aws_handler.save_checkpoint(
                    state, is_best=is_best, filename=f"checkpoint_epoch_{epoch}.pth.tar"
                )

                # Check if target accuracy reached
                if top1_acc >= 70.0:
                    logger.info(
                        f"\nTarget accuracy of 70% achieved! Current accuracy: {top1_acc:.2f}%"
                    )
                    message = f"Target accuracy reached: {top1_acc:.2f}%"
                    logger.info(message)
                    aws_handler.send_notification(message, "Training Target Achieved")
                    aws_handler.save_checkpoint(
                        state,
                        is_best=is_best,
                        filename=f"target_achieved_model_{epoch}.pth.tar",
                    )

                    break

            scheduler.step()

        # Final notification
        if args.local_rank <= 0:
            end_time = datetime.now()
            training_time = end_time - start_time
            message = f"Training completed. Best accuracy: {best_acc:.2f}%"
            aws_handler.send_notification(message, "Training Completed")
            logger.info("\nTraining completed!")
            logger.info(f"Total training time: {training_time}")
            aws_handler.save_tensorboard_logs(log_dir)
            aws_handler.cleanup()

    except Exception as e:
        if args.local_rank <= 0:
            error_message = f"Training failed: {str(e)}"
            logger.error(error_message)
            aws_handler.send_notification(error_message, "Training Error")
        raise e


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(0)
