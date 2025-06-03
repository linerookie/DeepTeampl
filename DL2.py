import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset as TorchDataset, DataLoader
import numpy as np

# Dataset classes (DL.py에서 가져온 데이터셋 클래스)
class Dataset:
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')

    def __len__(self):
        return len(self.data.get("filenames", []))

    def __getitem__(self, idx):
        return {
            'filenames': self.data['filenames'][idx],
            'fine_labels': self.data['fine_labels'][idx],
            'coarse_labels': self.data['coarse_labels'][idx],
            'data': self.data['data'][idx].reshape(3, 32, 32)
        }

# PyTorch Dataset wrapper (DL.py에서 가져온 래퍼 클래스)
class CIFAR100TorchDataset(TorchDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.__getitem__(idx)
        
        # 이미지 데이터 처리 (3, 32, 32) -> PIL Image 형태로 변환
        image = sample['data']  # (3, 32, 32) numpy array
        image = np.transpose(image, (1, 2, 0))  # (32, 32, 3)으로 변환
        
        if self.transform:
            image = self.transform(image)
        
        label = sample['fine_labels']
        
        return image, label

# 1. 데이터 전처리 및 로더 (DL.py의 transform을 기반으로 수정)
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
])

# 데이터셋 로딩 (DL.py 방식 사용)
print("CIFAR-100 데이터셋 로딩 중...")
train_dataset = Dataset("cifar-100-python/train")
test_dataset = Dataset("cifar-100-python/test")

# PyTorch Dataset wrapper 적용
train_dataset_torch = CIFAR100TorchDataset(train_dataset, transform=transform_train)
test_dataset_torch = CIFAR100TorchDataset(test_dataset, transform=transform_test)

# 디바이스 설정 (CUDA 강제 사용)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(f"사용 디바이스: {DEVICE}")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print(f"사용 디바이스: {DEVICE} (CUDA 사용 불가)")

trainloader = DataLoader(train_dataset_torch, batch_size=64, shuffle=True, num_workers=0)
testloader = DataLoader(test_dataset_torch, batch_size=64, shuffle=False, num_workers=0)

# 2. EfficientNet-B0 모델 준비
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 100)
model = model.to(DEVICE)

# 3. CutMix 함수
def rand_bbox(size, lam):
    W, H = size[3], size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bby1:bby2, bbx1:bbx2] = shuffled_data[:, :, bby1:bby2, bbx1:bbx2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size(-1) * data.size(-2)))
    return data, targets, shuffled_targets, lam

# 4. 학습 및 평가 루프
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(1, 31):  # 30 epochs (필요에 따라 조절)
    model.train()
    running_loss = 0.0
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        r = np.random.rand(1)
        if r < 0.5:
            inputs, targets1, targets2, lam = cutmix(inputs, targets)
            outputs = model(inputs)
            loss = lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch} | Loss: {running_loss/len(trainloader):.4f}")

    # 간단한 테스트 정확도 평가 (매 에폭)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print(f"Test accuracy: {100. * correct / total:.2f}%")

print("학습 및 평가 완료!")
