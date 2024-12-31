import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
from tqdm import tqdm
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
from datetime import timedelta, datetime
import logging
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import argparse
import math


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Training configuration
class Config:
    num_epochs = 150
    batch_size = 512
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


def get_data_loaders(subset_size=None, distributed=False, world_size=None, rank=None):
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Modified data augmentation for training
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # Removed interpolation and antialias
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.5),  # Moved after ToTensor
        ]
    )

    # Modified transform for validation
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),  # Removed antialias
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    training_folder_name = "ILSVRC/Data/CLS-LOC/train"
    val_folder_name = "ILSVRC/Data/CLS-LOC/val"

    train_dataset = torchvision.datasets.ImageFolder(
        root=training_folder_name, transform=train_transform
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=val_folder_name, transform=val_transform
    )

    # Create subset for initial testing
    if subset_size:
        train_indices = torch.randperm(len(train_dataset))[:subset_size]
        val_indices = torch.randperm(len(val_dataset))[: subset_size // 10]
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=(train_sampler is None),
        num_workers=Config.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    epoch_start_time = datetime.now()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_amp)

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
def setup_logging(log_dir):
    # Create local log directory
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


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
    stats = {
        "num_train_samples": len(train_loader.dataset),
        "num_val_samples": len(val_loader.dataset),
        "num_classes": len(train_loader.dataset.dataset.classes)
        if hasattr(train_loader.dataset, "dataset")
        else len(train_loader.dataset.classes),
        "batch_size": train_loader.batch_size,
        "num_train_batches": len(train_loader),
        "num_val_batches": len(val_loader),
        "device_count": torch.cuda.device_count(),
        "max_epochs": Config.num_epochs,
        "learning_rate": Config.learning_rate,
        "weight_decay": Config.weight_decay,
        "num_workers": Config.num_workers,
    }

    # Get class distribution
    if hasattr(train_loader.dataset, "dataset"):
        # For subset dataset
        classes = train_loader.dataset.dataset.classes
        class_to_idx = train_loader.dataset.dataset.class_to_idx
    else:
        # For full dataset
        classes = train_loader.dataset.classes
        class_to_idx = train_loader.dataset.class_to_idx

    stats["classes"] = classes
    stats["class_to_idx"] = class_to_idx

    return stats


# Modify the main function to support distributed training
def main():
    start_time = datetime.now()
    args = setup_distributed()

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",  # Use environment variables for initialization
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging and tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/resnet50_{timestamp}"
    if args.local_rank <= 0:  # Only create directories for master process
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        logger = setup_logging(log_dir)
        logger.info(f"Starting training on {socket.gethostname()}")
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"Training started at: {start_time}")

    set_seed()

    # Create model
    model = create_resnet50()
    if args.local_rank != -1:
        model = DDP(model.to(device), device_ids=[args.local_rank])
    else:
        model = torch.nn.DataParallel(model).to(device)

    # Rest of your training setup
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
                math.pi * (epoch - warmup_epochs) / (Config.num_epochs - warmup_epochs)
            )
        )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    # Get data loaders with distributed sampler
    train_loader, val_loader, train_sampler = get_data_loaders(
        subset_size=Config.subset_size,
        distributed=(args.local_rank != -1),
        world_size=dist.get_world_size() if args.local_rank != -1 else None,
        rank=args.local_rank if args.local_rank != -1 else None,
    )

    # Log dataset statistics
    if args.local_rank <= 0:
        dataset_stats = get_dataset_stats(train_loader, val_loader)
        logger.info("Dataset Statistics:")
        logger.info(f"Training samples: {dataset_stats['num_train_samples']}")
        logger.info(f"Validation samples: {dataset_stats['num_val_samples']}")
        logger.info(f"Number of classes: {dataset_stats['num_classes']}")
        logger.info(f"Batch size: {dataset_stats['batch_size']}")
        logger.info(f"Training batches per epoch: {dataset_stats['num_train_batches']}")
        logger.info(f"Validation batches per epoch: {dataset_stats['num_val_batches']}")

    best_acc = 0
    # Training loop
    total_training_time = timedelta()
    # Training loop
    for epoch in range(Config.num_epochs):
        if args.local_rank <= 0:
            logger.info(f"Starting epoch {epoch}")

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train for one epoch and get metrics
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device
        )
        total_training_time += train_metrics["time"]

        # train_epoch(model, train_loader, criterion, optimizer, epoch, device)

        if args.local_rank <= 0:  # Only validate on master process
            # Log training metrics
            logger.info(
                f"Epoch {epoch} completed in {train_metrics['time']}, "
                f"Training Loss: {train_metrics['loss']:.4f}, "
                f"Training Accuracy: {train_metrics['accuracy']:.2f}%"
            )

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "top1_accuracy": top1_acc,
                    "top5_accuracy": top5_acc,
                },
                os.path.join(log_dir, "best_model.pth"),
            )

            if top1_acc >= 70.0:
                logger.info(
                    f"\nTarget accuracy of 70% achieved! Current accuracy: {top1_acc:.2f}%"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_acc": best_acc,
                        "top1_accuracy": top1_acc,
                        "top5_accuracy": top5_acc,
                    },
                    os.path.join(log_dir, "target_achieved_model.pth"),
                )
                # break

        # Save metrics after each epoch
        # with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        #     json.dump(train_metrics, f, indent=4)

        scheduler.step()

    if args.local_rank <= 0:
        end_time = datetime.now()
        training_time = end_time - start_time
        writer.close()
        logger.info("\nTraining completed!")
        logger.info(f"Total training time: {training_time}")
        logger.info(f"Best Top-1 Accuracy: {train_metrics['best_top1_acc']:.2f}%")
        logger.info(
            f"Target accuracy of 70% {'achieved' if train_metrics['best_top1_acc'] >= 70.0 else 'not achieved'}"
        )


if __name__ == "__main__":
    main()
