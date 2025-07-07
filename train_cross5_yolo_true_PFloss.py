import warnings

warnings.filterwarnings("ignore")
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from trajectory_extraction import extract_single_trajectory_by_date
from model import CNNTransformer_ResNet101_Fusion, CNNTransformer_EfficientNet_Fusion
from tqdm import tqdm
import gc
from collections import Counter


# -------- 配置 --------
label_csv_path = './fusion_labels_stratified.csv'
log_dir = 'logs_tensorboard_fusion_full_epoch20250703'
batch_size = 32
num_epochs = 200
input_dim = 9
image_size = 224
num_classes = 5
# chunk_size = 200000 # 每个子集大小186,656
chunk_size = 186656 # 每个子集大小186,656
num_workers_max = min(os.cpu_count(), 20)

# -------- 环境设置 --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=log_dir)
class_map = {'Cargo': 0, 'Fishing': 1, 'Tanker': 2, 'Passenger': 3, 'Military': 4}
log_txt_path = os.path.join(log_dir, "training_log.txt")
os.makedirs(log_dir, exist_ok=True)
with open(log_txt_path, 'w') as f:
    f.write("epoch\tloss\tacc\tf1\tprecision\trecall\n")

class PolyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, poly_alpha=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.poly_alpha = poly_alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        pt = torch.clamp(pt, min=1e-6, max=1.0)  # 防止不稳定

        focal_term = (1 - pt) ** self.gamma
        poly_term = self.poly_alpha * (1 - pt) ** 2
        loss = focal_term * ce_loss + poly_term
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# -------- Dataset --------
class FusionLazyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        max_retry = 10
        for _ in range(max_retry):
            row = self.df.iloc[idx]
            traj = extract_single_trajectory_by_date(row['data_path'], row['image_path'])
            if traj is None or len(traj) == 0 or (isinstance(traj, np.ndarray) and traj.shape[0] == 0):
                idx = np.random.randint(0, len(self.df))
                continue



            traj = torch.tensor(traj, dtype=torch.float32)

            # MAX_SEQ_LEN = 1440
            MAX_SEQ_LEN = 7200

            if traj.shape[0] > MAX_SEQ_LEN:
                traj = traj[:MAX_SEQ_LEN]

            img = Image.open(row['image_path']).convert('RGB')
            img = self.transform(img) if self.transform else img
            label = class_map[row['label']]
            return traj, img, label
        raise ValueError(f"⚠️ 多次抽样无轨迹数据: {row['image_path']}")

# -------- Collate --------
def fusion_collate_fn(batch):
    batch = [(t, i, l) for t, i, l in batch if t.shape[0] > 0]
    if len(batch) == 0:
        raise ValueError("❌ 本批次全部轨迹为空")
    traj, img, label = zip(*batch)
    lengths = [len(x) for x in traj]
    max_len = max(lengths)
    padded = torch.zeros(len(batch), max_len, traj[0].shape[1])
    mask = torch.zeros(len(batch), max_len)
    for i, seq in enumerate(traj):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1
    return padded, torch.stack(img), torch.tensor(label), mask, torch.tensor(lengths)

# -------- 划分子集 --------
def split_dataframe_by_size(df, chunk_size):
    return [df.iloc[i:i+chunk_size].reset_index(drop=True) for i in range(0, len(df), chunk_size)]

# -------- 模型训练与评估 --------
def train_model(model, full_train_df, test_loader, transform, criterion, optimizer, epochs=50, patience=10, chunk_size=2000):
    best_f1, no_improve = 0, 0
    train_chunks = split_dataframe_by_size(full_train_df, chunk_size)

    for epoch in range(epochs):
        chunk_idx = epoch % len(train_chunks)
        current_chunk = train_chunks[chunk_idx]
        train_dataset = FusionLazyDataset(current_chunk, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate_fn, num_workers=num_workers_max,persistent_workers=True)

        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Training [Chunk {chunk_idx+1}/{len(train_chunks)}]", ncols=100)
        for traj, img, label, mask, lengths in pbar:
            traj, img, label, mask = traj.to(device), img.to(device), label.to(device), mask.to(device)
            optimizer.zero_grad()
            out = model(traj, lengths, img)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        acc, f1, prec, rec = evaluate_model(model, test_loader, epoch)

        writer.add_scalar('Loss/Train', avg_loss, epoch)
        with open(log_txt_path, 'a') as f:
            f.write(f"{epoch+1}\t{avg_loss:.4f}\t{acc:.4f}\t{f1:.4f}\t{prec:.4f}\t{rec:.4f}\n")

        if f1 > best_f1 + 1e-4:
            best_f1 = f1
            no_improve = 0
            torch.save(model.state_dict(), 'best_fusion_model.pth')
            print(f"✅ Best model saved at Epoch {epoch+1} (F1={f1:.4f})")
        else:
            no_improve += 1
            print(f"⏳ No improvement. Patience: {no_improve}/{patience}")
            if no_improve >= patience:
                print("🛑 Early stopping triggered.")
                break

# -------- 评估函数 --------
def evaluate_model(model, loader, epoch=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[Epoch {epoch}] Evaluating", ncols=100)
        for traj, img, label, mask, lengths in pbar:
            traj, img, label, mask = traj.to(device), img.to(device), label.to(device), mask.to(device)
            out = model(traj, lengths, img)
            preds = torch.argmax(out, dim=1)
            y_true.extend(label.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted')

    if epoch is not None:
        writer.add_scalar("F1/Test", f1, epoch)
        writer.add_scalar("Acc/Test", acc, epoch)
        writer.add_scalar("Precision/Test", precision, epoch)
        writer.add_scalar("Recall/Test", recall, epoch)

        # # ✅ 追加记录验证指标到日志文件
        # with open(log_txt_path, 'a') as f:
        #     f.write(
        #         f"[Epoch {epoch}]\tACC={acc:.4f}\tF1={f1:.4f}\tPrecision={precision:.4f}\tRecall={recall:.4f}\n")

    print(f"[Epoch {epoch}]: Acc={acc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    return acc, f1, precision, recall

# -------- 主程序 --------
if __name__ == '__main__':
    df = pd.read_csv(label_csv_path)
    test_df = df[df['split'] == 'test']
    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    class_order = ['Cargo', 'Fishing', 'Tanker', 'Passenger', 'Military']
    label_counter = Counter(train_df['label'])
    counts = [label_counter[c] for c in class_order]
    weights = torch.tensor([1.0 / c for c in counts], dtype=torch.float32)
    weights = weights / weights.sum()
    weights = weights.to(device)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = FusionLazyDataset(test_df, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=fusion_collate_fn, num_workers=num_workers_max,persistent_workers=True)

    model = CNNTransformer_EfficientNet_Fusion(input_dim=input_dim, num_classes=num_classes)
    if torch.cuda.device_count() > 1:
        print(f"✅ 检测到 {torch.cuda.device_count()} 张GPU，启用 DataParallel")
        model = nn.DataParallel(model)
    model = model.to(device)

    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # ✅ 推荐

    criterion = PolyFocalLoss(alpha=weights, gamma=2.0, poly_alpha=1.0)

    criterion = nn.CrossEntropyLoss()

    print(f"\n🚀 开始第训练阶段...")
    train_model(model, train_df, test_loader, transform, criterion, optimizer, epochs=num_epochs, patience=10, chunk_size=chunk_size)

    print("\n>>> 最终评估")
    model.load_state_dict(torch.load('best_fusion_model_P.pth'))
    evaluate_model(model, test_loader, epoch=999)
    torch.cuda.empty_cache()
    gc.collect()
    writer.close()
