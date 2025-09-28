import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from math_transformer import MathTransformer
from math_tokenizer import MathTokenizer
from data_generator import MathDataGenerator
from expression_evaluator import ExpressionEvaluator
from value_storage import ValueStorage

class MathDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        full_sequence = sample['full_sequence']

        input_ids = self.tokenizer.encode(full_sequence, max_length=self.max_length)
        attention_mask = self.tokenizer.create_attention_mask(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

class Trainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = self._setup_device()
        self.tokenizer = MathTokenizer()
        self.model = self._create_model()
        self.optimizer = None
        self.scaler = GradScaler() if self.config['device']['mixed_precision'] else None
        self.writer = SummaryWriter(f"runs/math_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.storage = ValueStorage("model_values.db")

    def _setup_device(self):
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['device_id']}")
            print(f"Using GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def _create_model(self):
        model = MathTransformer(
            vocab_size=self.config['model']['vocab_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            num_heads=self.config['model']['num_attention_heads'],
            max_position_embeddings=self.config['model']['max_position_embeddings'],
            dropout=self.config['model']['dropout']
        )
        model = model.to(self.device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        return model

    def prepare_data(self):
        generator = MathDataGenerator()

        print("Generating training data...")
        train_data = generator.generate_training_data(self.config['data']['num_training_samples'])

        print("Generating validation data...")
        val_data = generator.generate_training_data(self.config['data']['num_validation_samples'])

        train_dataset = MathDataset(
            train_data,
            self.tokenizer,
            self.config['data']['max_expression_length']
        )

        val_dataset = MathDataset(
            val_data,
            self.tokenizer,
            self.config['data']['max_expression_length']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            labels[labels == self.tokenizer.special_tokens['<PAD>']] = -100

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids, attention_mask)
                    logits = outputs['logits']
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                loss = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

            global_step = epoch * len(train_loader) + batch_idx
            if global_step % self.config['training']['logging_steps'] == 0:
                self.writer.add_scalar('Loss/train', loss.item(), global_step)

        return total_loss / len(train_loader)

    @torch.no_grad()
    def evaluate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            labels[labels == self.tokenizer.special_tokens['<PAD>']] = -100

            outputs = self.model(input_ids, attention_mask)
            logits = outputs['logits']
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Loss/val', avg_loss, epoch)

        return avg_loss

    def train(self):
        train_loader, val_loader = self.prepare_data()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )

        best_val_loss = float('inf')
        Path("checkpoints").mkdir(exist_ok=True)

        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")

            train_loss = self.train_epoch(train_loader, epoch)
            print(f"Training Loss: {train_loss:.4f}")

            if epoch % self.config['training']['eval_steps'] == 0:
                val_loss = self.evaluate(val_loader, epoch)
                print(f"Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True)

                self.storage.set(f"epoch_{epoch}_val_loss", val_loss)

            scheduler.step()
            self.writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)

            if epoch % self.config['training']['save_steps'] == 0:
                self.save_checkpoint(epoch, train_loss)

        print("\nTraining completed!")
        self.writer.close()

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config,
            'tokenizer_variables': self.tokenizer.variables,
        }

        if is_best:
            path = Path("checkpoints/best_model.pt")
        else:
            path = Path(f"checkpoints/checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'tokenizer_variables' in checkpoint:
            self.tokenizer.variables = checkpoint['tokenizer_variables']
        print(f"Loaded checkpoint from {checkpoint_path}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()