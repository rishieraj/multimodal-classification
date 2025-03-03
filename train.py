# train.py

import os
import os.path as osp
import argparse
import yaml
import json
import matplotlib
matplotlib.use('Agg')

from easydict import EasyDict as edict
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from sklearn.metrics import precision_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.meters import AverageMeter
from data.dataset import MaterialDataset, ContrastiveDataset
from torch.utils.data import DataLoader

class MaterialTrainer:
    """Training and evaluation framework for material classification
    
    This class handles:
    1. Single modality classification (Task 1)
    2. Multi-modal fusion (Task 2)
    3. Contrastive learning (Task 3)
    """
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        self.setup_environment()
        self.build_dataloaders()
        self.build_model()
        self.setup_optimization()
        self.setup_logging()
        self.train_losses = []
        self.train_accs = []
        self.train_maps = []
        self.val_losses = []
        self.val_accs = []
        self.val_maps = []

    def setup_environment(self):
        """Setup random seeds and computing device"""
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def build_dataloaders(self):
        """Initialize train/val/test dataloaders"""
        if self.args.task == 'contrastive':
            datasets = {
                split: ContrastiveDataset(
                    self.args,
                    split,
                ) for split in ['train', 'val', 'test']
            }
        else:
            datasets = {
                split: MaterialDataset(
                    self.args,
                    split,
                ) for split in ['train', 'val', 'test']
            }
        
        self.dataloaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=datasets['train'].collate_fn
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=datasets['val'].collate_fn
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=datasets['test'].collate_fn
            )
        }

    def build_model(self):
        """Initialize model based on task type"""
        if self.args.task == 'unimodal':
            from models.unimodal import VisionClassifier, TouchClassifier, AudioClassifier
            model_classes = {
                'vision': VisionClassifier,
                'touch': TouchClassifier,
                'audio': AudioClassifier
            }
            self.model = model_classes[self.args.modality](
                num_classes=self.cfg.num_classes,
                backbone=self.args.backbone,
            )
        elif self.args.task == 'multimodal':
            from models.multimodal import LateFusion, AttentionFusion
            if self.args.fusion_type == 'late':
                self.model = LateFusion(
                    input_dims=None,  # Not needed for this implementation
                    num_classes=self.cfg.num_classes
                )
            else:  # attention
                self.model = AttentionFusion(
                    input_dims=None,  # Not needed for this implementation
                    num_heads=4,
                    num_classes=self.cfg.num_classes
                )
        elif self.args.task == 'contrastive':
            from models.contrastive import ContrastiveLearning
            self.model = ContrastiveLearning(
                modalities=self.args.modality_list,
                feature_dim=128,  # Can be configured through parameters
                num_classes=self.cfg.num_classes_contrastive
            )
        
        self.model = self.model.to(self.device)

    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler"""
        if self.args.finetune:
            # Different learning rates for pretrained layers and new layers
            backbone_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if 'classifier' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            
            self.optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': self.args.lr * 0.1},
                {'params': head_params, 'lr': self.args.lr}
            ], weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6
        )

    def setup_logging(self):
        """Setup experiment logging"""
        self.exp_dir = osp.join('./experiments', self.args.task, self.args.exp)
        self.ckpt_dir = osp.join(self.exp_dir, 'checkpoints')  # add checkpoints directory
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)  # create checkpoints directory
        self.best_acc = 0.0
        self.best_loss = float('inf')

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        if self.args.task == 'contrastive':
            # contrastive learning training loop
            for batch in self.dataloaders['train']:
                # move data to device
                for modality in self.args.modality_list:
                    batch[f'{modality}_1'] = batch[f'{modality}_1'].to(self.device)
                    batch[f'{modality}_2'] = batch[f'{modality}_2'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                
                # forward propagation
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # backward propagation
                loss.backward()
                self.optimizer.step()
                
                # only record loss
                running_loss += loss.item()
        else:
            # original classification training loop
            for batch in self.dataloaders['train']:
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs['loss']
                preds = outputs['pred']
                
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                all_preds.append(preds.detach())
                all_labels.append(batch['label'])
        
        # calculate average loss
        epoch_loss = running_loss / len(self.dataloaders['train'])
        
        if self.args.task == 'contrastive':
            # contrastive learning only record loss
            self.train_losses.append(epoch_loss)
            accuracy = 0
            mAP = 0
        else:
            # classification task calculate all metrics
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            label_one_hot = torch.zeros(all_labels.size(0), self.cfg.num_classes)
            label_one_hot.scatter_(1, all_labels.cpu().unsqueeze(1), 1)
            
            accuracy = (all_preds.argmax(dim=1) == all_labels).float().mean() * 100
            mAP = average_precision_score(
                label_one_hot.numpy(),
                all_preds.cpu().numpy()
            ) * 100
            
            self.train_losses.append(epoch_loss)
            self.train_accs.append(accuracy.item())
            self.train_maps.append(mAP)
        
        return epoch_loss

    @torch.no_grad()
    def evaluate(self, split='val'):
        """Evaluate model on validation or test set"""
        self.model.eval()
        epoch_loss = AverageMeter()
        
        if self.args.task == 'contrastive':
            # contrastive learning only calculate loss
            for batch in tqdm(self.dataloaders[split], desc=f'Evaluate {split}'):
                # Move data to device
                for modality in self.args.modality_list:
                    batch[f'{modality}_1'] = batch[f'{modality}_1'].to(self.device)
                    batch[f'{modality}_2'] = batch[f'{modality}_2'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                loss = outputs['loss']
                
                # Update loss
                epoch_loss.update(loss.item(), batch['label'].size(0))
            
            metrics = {
                'loss': epoch_loss.avg,
                'accuracy': 0, 
                'mAP': 0 
            }
            
            if split == 'val':
                self.val_losses.append(metrics['loss'])
            
        else:
            # original classification evaluation code
            all_preds = []
            all_labels = []
            
            for batch in tqdm(self.dataloaders[split], desc=f'Evaluate {split}'):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(batch)
                loss = outputs['loss']
                preds = outputs['pred']
                
                epoch_loss.update(loss.item(), batch['label'].size(0))
                all_preds.append(preds)
                all_labels.append(batch['label'])
            
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            label_one_hot = torch.zeros(all_labels.size(0), self.cfg.num_classes)
            label_one_hot.scatter_(1, all_labels.cpu().unsqueeze(1), 1)
            
            accuracy = (all_preds.argmax(dim=1) == all_labels).float().mean() * 100
            mAP = average_precision_score(
                label_one_hot.numpy(),
                all_preds.cpu().numpy()
            ) * 100
            
            metrics = {
                'loss': epoch_loss.avg,
                'accuracy': accuracy.item(),
                'mAP': mAP
            }
            
            if split == 'val':
                self.val_losses.append(metrics['loss'])
                self.val_accs.append(metrics['accuracy'])
                self.val_maps.append(metrics['mAP'])
        
        return metrics

    def save_checkpoint(self, metrics, epoch):
        """Save model checkpoint"""
        # Save checkpoint for current epoch
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': self.args,
            'cfg': self.cfg
        }
        
        # Save checkpoint every N epochs
        if epoch % 5 == 0:
            torch.save(
                checkpoint,
                osp.join(self.ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Also save the best model
        if metrics['accuracy'] > self.best_acc:
            self.best_acc = metrics['accuracy']
            torch.save(
                checkpoint,
                osp.join(self.exp_dir, 'best_model.pth')
            )

    def save_visualizations(self, all_preds, all_labels, split):
        """Save visualization plots
        
        For contrastive learning:
        - Only save loss curves
        
        For classification tasks:
        - Save confusion matrix
        - Save training curves (loss, accuracy, mAP)
        - Save per-class accuracy
        """
        vis_dir = osp.join(self.exp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Training curves
        if len(self.train_losses) > 0:
            if self.args.task == 'contrastive':
                # For contrastive learning, only plot loss curves
                plt.figure(figsize=(8, 6))
                plt.plot(self.train_losses, label='Train')
                plt.plot(self.val_losses, label='Val')
                plt.title('Contrastive Loss Curves')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.tight_layout()
                plt.savefig(osp.join(vis_dir, 'training_curves.png'))
                plt.close()
                return  # Early return for contrastive learning
            
            # For classification tasks, plot all metrics
            plt.figure(figsize=(15, 5))
            
            # Loss curves
            plt.subplot(1, 3, 1)
            plt.plot(self.train_losses, label='Train')
            plt.plot(self.val_losses, label='Val')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Accuracy curves
            plt.subplot(1, 3, 2)
            plt.plot(self.train_accs, label='Train')
            plt.plot(self.val_accs, label='Val')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            
            # mAP curves
            plt.subplot(1, 3, 3)
            plt.plot(self.train_maps, label='Train')
            plt.plot(self.val_maps, label='Val')
            plt.title('mAP')
            plt.xlabel('Epoch')
            plt.ylabel('mAP (%)')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(osp.join(vis_dir, 'training_curves.png'))
            plt.close()
            
            # Confusion matrix and class accuracy only for classification tasks
            pred_labels = all_preds.argmax(dim=1).cpu().numpy()
            true_labels = all_labels.cpu().numpy()
            cm = confusion_matrix(true_labels, pred_labels)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix ({split})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(osp.join(vis_dir, f'{split}_confusion_matrix.png'))
            plt.close()
            
            # Per-class accuracy
            num_classes = self.cfg.num_classes
            class_correct = np.zeros(num_classes)
            class_total = np.zeros(num_classes)
            for pred, true in zip(pred_labels, true_labels):
                class_total[true] += 1
                if pred == true:
                    class_correct[true] += 1
            
            class_acc = class_correct / (class_total + 1e-8) * 100
            class_results = pd.DataFrame({
                'Class': range(num_classes),
                'Accuracy': class_acc,
                'Samples': class_total
            })
            class_results.to_csv(osp.join(vis_dir, f'{split}_class_accuracy.csv'))

    def train(self):
        """Main training loop"""
        for epoch in range(self.args.epochs):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.evaluate('val')
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            self.save_checkpoint(val_metrics, epoch)
            
            # Save visualizations after each epoch
            if self.args.task == 'contrastive':
                self.save_visualizations(None, None, 'train')  # Contrastive learning does not need predictions or labels
            else:
                # Get latest predictions and labels for visualization
                with torch.no_grad():
                    all_preds = []
                    all_labels = []
                    for batch in self.dataloaders['train']:
                        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                                for k, v in batch.items()}
                        outputs = self.model(batch)
                        all_preds.append(outputs['pred'].detach())
                        all_labels.append(batch['label'])
                    
                    all_preds = torch.cat(all_preds, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    self.save_visualizations(all_preds, all_labels, 'train')
            
            # Logging
            if self.args.task == 'contrastive':
                print(f"Epoch {epoch}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
            else:
                print(f"Epoch {epoch}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Acc: {val_metrics['accuracy']:.2f}")
                print(f"Val mAP: {val_metrics['mAP']:.2f}")

    def test(self):
        """Evaluate on test set"""
        checkpoint = torch.load(
            osp.join(self.exp_dir, 'best_model.pth'),
            weights_only=False
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # run evaluation and generate visualizations
        test_metrics = self.evaluate('test')
        
        # save results
        results_path = osp.join(self.exp_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print("Test Results:")
        print(f"Loss: {test_metrics['loss']:.4f}")
        print(f"Accuracy: {test_metrics['accuracy']:.2f}")
        print(f"mAP: {test_metrics['mAP']:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    # Task
    parser.add_argument('--task', type=str, required=True,
                      choices=['unimodal', 'multimodal', 'contrastive'])
    parser.add_argument('--modality', type=str, default='vision',
                      choices=['vision', 'touch', 'audio'])
    parser.add_argument('--modality_list', nargs='+', default=['vision', 'touch', 'audio'],
                      help='List of modalities to use')
    parser.add_argument('--fusion_type', type=str, default='late',
                      choices=['early', 'late', 'attention'])
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet',
                      choices=['resnet', 'fenet'])
    parser.add_argument('--pretrained_path', type=str, default=None)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Paths
    parser.add_argument('--data_location', type=str, required=True)
    parser.add_argument('--config_location', type=str, required=True)
    parser.add_argument('--split_location', type=str, required=True)
    parser.add_argument('--exp', type=str, default='default')
    
    # Mode
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--finetune', action='store_true')
    
    # Contrastive learning parameters
    parser.add_argument('--temperature', type=float, default=0.07,
                      help='Temperature parameter for contrastive loss')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return edict(config)

def main():
    # Parse arguments and load config
    args = parse_args()
    cfg = load_config(args.config_location)
    
    # Load and display dataset split statistics
    print("\nLoading dataset splits...")
    with open(args.split_location, 'r') as f:
        splits = json.load(f)

    # Calculate size for each split
    train_size = len(splits['train'])
    val_size = len(splits['val'])
    test_size = len(splits['test'])
    total_size = train_size + val_size + test_size

    print(f"\nDataset split statistics:")
    print(f"Train set: {train_size} samples ({train_size/total_size*100:.1f}%)")
    print(f"Val set:   {val_size} samples ({val_size/total_size*100:.1f}%)")
    print(f"Test set:  {test_size} samples ({test_size/total_size*100:.1f}%)")
    print(f"Total:     {total_size} samples")
    print("-" * 50)
    
    # Initialize trainer
    trainer = MaterialTrainer(args, cfg)
    
    # Run training or evaluation
    if not args.eval:
        trainer.train()
    trainer.test()

if __name__ == '__main__':
    main()