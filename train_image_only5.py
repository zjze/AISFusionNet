#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : AIS多模态
@File ：train_image_only2.py
@Auth ： zjz
@Data ： 2025/5/15 10:27
Note:按照MMSI进行8:2划分
5个类别

EfficientNet:Acc=0.7604, F1=0.7588, Precision=0.7589, Recall=0.7604
MobileNetV2:Acc=0.7434, F1=0.7422, Precision=0.7421, Recall=0.7434
Resnet101:Acc=0.8149, F1=0.8140, Precision=0.8136, Recall=0.8149

"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 配置
image_dir = r'./images'
log_dir = r'logs_tensorboard_image5'
image_size = 224
batch_size = 32
num_epochs = 100
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Cargo', 'Fishing', 'Tanker', 'Passenger', 'Military']
label_map = {name: i for i, name in enumerate(class_names)}
writer = SummaryWriter(log_dir=log_dir)

# 数据集定义
class ShipImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# 数据加载（按类别8:2划分）
def load_image_data_stratified(image_dir):
    image_paths, labels = [], []

    for file in os.listdir(image_dir):
        if file.endswith('.jpg'):
            parts = file.split('_')
            if len(parts) < 3:
                continue
            label_str = parts[-1].replace('.jpg', '')
            if label_str not in label_map:
                continue
            image_paths.append(os.path.join(image_dir, file))
            labels.append(label_map[label_str])

    # 分类别划分
    class_images = defaultdict(list)
    class_labels = defaultdict(list)

    for path, label in zip(image_paths, labels):
        class_images[label].append(path)
        class_labels[label].append(label)

    X_train, X_test, y_train, y_test = [], [], [], []
    for label in class_images:
        X_tr, X_te, y_tr, y_te = train_test_split(
            class_images[label], class_labels[label], test_size=0.2, random_state=42
        )
        X_train.extend(X_tr)
        y_train.extend(y_tr)
        X_test.extend(X_te)
        y_test.extend(y_te)

    return X_train, X_test, y_train, y_test

# EfficientNet 模型
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes, weight_path=None):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = EfficientNet.from_name('efficientnet-b0')

        # 如果指定了本地权重路径，则加载预训练权重
        if weight_path and os.path.exists(weight_path):
            state_dict = torch.load(weight_path)
            self.backbone.load_state_dict(state_dict)
            print(f"✅ Loaded EfficientNet weights from {weight_path}")
        else:
            print("⚠️ 未加载预训练权重，模型将从头训练")



        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 模型训练
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs, patience=15):
    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/Train', avg_loss, epoch)

        acc, f1 = evaluate_model(model, test_loader, epoch)

        print(
            f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f} ")

        if f1 > best_f1 + 1e-4:  # 防止浮点数误差影响判断
            best_f1 = f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_image_model.pth')
            print(f"✅ New best model saved at epoch {epoch+1} with F1={f1:.4f}")
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1} complete. Loss={avg_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}, No improvement for {epochs_no_improve} epochs")

        # 如果超过早停轮数，则终止训练
        if epochs_no_improve >= patience:
            print(f"⛔ Early stopping at epoch {epoch+1} due to no improvement in F1 score for {patience} epochs.")
            break


# 模型评估
def evaluate_model(model, data_loader, epoch=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')

    if epoch is not None and isinstance(epoch, int):
        writer.add_scalar('Acc/Test', acc, epoch)
        writer.add_scalar('F1/Test', f1, epoch)
        writer.add_scalar('Precision/Test', precision, epoch)
        writer.add_scalar('Recall/Test', recall, epoch)

    print(f"[Epoch {epoch}] Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    return acc, f1

# 入口
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    X_train, X_test, y_train, y_test = load_image_data_stratified(image_dir)

    train_loader = DataLoader(
        ShipImageDataset(X_train, y_train, transform), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        ShipImageDataset(X_test, y_test, transform), batch_size=batch_size, shuffle=False)

    # model = EfficientNetClassifier(num_classes=num_classes, weight_path='./efficientnet-b0-355c32eb.pth').to(device)
    model = EfficientNetClassifier(num_classes=num_classes, weight_path=' ').to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs)

    print("\n>>> 加载最优模型进行最终评估...")
    model.load_state_dict(torch.load('best_image_model.pth'))
    evaluate_model(model, test_loader, epoch=999)

    writer.close()

