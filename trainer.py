import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr=1e-3,
        weight_decay=5e-4,
        device="cuda",
        save_path="models/best_model.pt",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_path = save_path

        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.patience = 10
        self.counter = 0

        self.scaler = torch.amp.GradScaler("cuda")

        torch.backends.cudnn.benchmark = True
        self.epochs_times = []

        self.best_val_loss = float("inf")

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for x, y, _, _ in pbar:
            x = x.to(self.device)
            y = y.to(self.device).float()

            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", enabled=True):
                preds = self.model(x)
                loss = self.criterion(preds, y)

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y, _, _ in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device).float()

                preds = self.model(x)
                loss = self.criterion(preds, y)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0

                torch.save(self.model.state_dict(), self.save_path)
                print("--------- Saved best model")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("--------- Early stopping")
                    break

            self.scheduler.step(val_loss)
