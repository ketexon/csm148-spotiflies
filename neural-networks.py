# %%
# imports
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import lightning as pl

from torch.utils.data import DataLoader, Dataset, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# %%
# import data
from spotify_data import grouped_data, categorical_columns, string_columns

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
response_variable = "track_genre"

# remove string columns and one-hot encode
Y = grouped_data[response_variable]
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

X = grouped_data.drop(columns=[response_variable, *string_columns])
X = pd.get_dummies(
    X,
    columns=[col for col in categorical_columns if col != response_variable],
    drop_first=True
)

# split data
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y_encoded,
    test_size=0.2,
    random_state=42
)

# scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

# create dataset
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
validation_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

# create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
)
validation_loader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False,
)


# %%
# Create model

class SpotifyModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr: float):
        super().__init__()
        self.lr = lr
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1),
        )
        self.epoch_metrics = []
        self.train_metrics_stack = []
        self.validation_metrics_stack = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics_stack.append({
            'train_loss': loss,
        })
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.validation_metrics_stack.append({
            'val_loss': loss,
            'val_acc': acc
        })
        return loss

    def on_validation_epoch_end(self):
        # sum in validation stack
        if len(self.validation_metrics_stack) == 0:
            return
        if len(self.train_metrics_stack) == 0:
            return
        val_loss = torch.stack([x['val_loss'] for x in self.validation_metrics_stack]).mean()
        val_acc = torch.stack([x['val_acc'] for x in self.validation_metrics_stack]).mean()
        train_loss = torch.stack([x['train_loss'] for x in self.train_metrics_stack]).mean()
        self.epoch_metrics.append({
            'val_loss': val_loss.cpu(),
            'val_acc': val_acc.cpu(),
            'train_loss': train_loss.cpu(),
        })
        # clear stack
        self.validation_metrics_stack = []
        self.train_metrics_stack = []


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


# %%
input_dim = X_train.shape[1]
output_dim = len(label_encoder.classes_)
lr = 0.01

model = SpotifyModel(input_dim, output_dim, lr)

# train model
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_loader, validation_loader)

# %%
# evaluate model
model.eval()
with torch.no_grad():
    outputs = model(X_val_tensor).cpu()
    _, predicted = torch.max(outputs.data, 1)
    class_report = classification_report(
        Y_val,
        predicted,
        target_names=label_encoder.classes_,
        zero_division=1
    )

print(class_report)

# %%
# Graph epoch metrics
import matplotlib.pyplot as plt

# plot epoch to val loss and train loss
plt.plot([x['val_loss'].cpu() for x in model.epoch_metrics], label='val_loss')
plt.plot([x['train_loss'].cpu() for x in model.epoch_metrics], label='train_loss')
plt.legend()
plt.show()
