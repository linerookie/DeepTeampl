import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import os

#사이즈 조절
BATCH_SIZE = 64  
LEARNING_RATE = 0.001 
EPOCHS = 100     
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALIDATION_SPLIT = 0.1  # 10% of training data for validation

# 시각화 설정
SHOW_SAMPLE_IMAGE = False     # 첫 번째 샘플 이미지 표시 안함
SHOW_MISCLASSIFICATIONS = False # 오분류 이미지 분석 표시 안함
SHOW_RESULT_PLOTS = False     # 결과 그래프 표시 안함

# CIFAR-100 class names - 자동으로 가져오기
def get_cifar100_classes():
    """CIFAR-100 클래스 이름을 자동으로 가져오는 함수"""
    try:
        # torchvision에서 CIFAR100 데이터셋 사용해서 클래스 이름 가져오기
        import torchvision.datasets as datasets
        
        # 임시로 CIFAR100 데이터셋 객체 생성 (다운로드 안함)
        temp_dataset = datasets.CIFAR100(root='./temp', train=True, download=False)
        if hasattr(temp_dataset, 'classes'):
            return temp_dataset.classes
    except:
        pass
    
    # 백업: 직접 정의된 클래스 이름들
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppies', 'porcupine', 
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
        'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
    ]

# 클래스 이름 가져오기 (선택사항)
try:
    CIFAR100_CLASSES = get_cifar100_classes()
    print(f"CIFAR-100 클래스 이름 로드 완료: {len(CIFAR100_CLASSES)}개 클래스")
except:
    CIFAR100_CLASSES = [f"Class_{i}" for i in range(100)]  # 간단한 대체
    print("클래스 이름 대신 번호 사용: Class_0, Class_1, ...")

# ResNet Implementation (사전학습 모델 기반)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # SE Block 추가
        self.use_se = use_se
        if use_se:
            self.se = SEBlock(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        
        self.dropout = nn.Dropout(0.1)  # Dropout 추가

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += self.shortcut(x)
        out = F.relu(out)
        out = self.dropout(out)  # Dropout 적용
        return out

# SE Block 추가
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, use_se=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.use_se = use_se

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # Global Average Pooling 대신 Adaptive Average Pooling 사용
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.use_se))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def SEResNet18():
    """SE-ResNet18 with Squeeze-and-Excitation blocks"""
    return ResNet(BasicBlock, [2, 2, 2, 2], use_se=True)

# EfficientNet Implementation
# Mixup 구현
def mixup_data(x, y, alpha=1.0):
    """Mixup 데이터 증강"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss 계산"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# PyTorch Dataset wrapper for your existing data
class CIFAR100TorchDataset(TorchDataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = data['data']  # Already in (3, 32, 32) format
        image = np.transpose(image, (1, 2, 0))  # Convert to HWC for transforms
        label = data['fine_labels']
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
            
        return image, label

# Enhanced Data transforms with more augmentation techniques
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    # CIFAR-100을 위한 추가 augmentation
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 더 강한 RandomErasing
])

val_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

def get_model_info(model):
    """모델의 파라미터 수와 크기 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모델 크기 추정 (MB)
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }

# Warm-up learning rate scheduler
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Warmup phase
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing phase
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def train_model_enhanced(model, train_loader, val_loader, test_loader, model_name, epochs=EPOCHS):
    """기본 향상된 훈련 함수 - validation 포함"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    best_val_acc = 0
    patience = 15
    patience_counter = 0
    
    # History tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': []
    }
    
    print(f"\nTraining {model_name}...")
    print(f"Device: {DEVICE}")
    model_info = get_model_info(model)
    print(f"Model Parameters: {model_info['total_params']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc = evaluate_model_detailed(model, val_loader, criterion)
        
        # Test (for monitoring)
        test_acc = evaluate_model(model, test_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_name.lower()}_cifar100.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history, model_info

def train_model_advanced(model, train_loader, val_loader, test_loader, model_name, epochs=EPOCHS, use_mixup=True, use_label_smoothing=True):
    """고급 최적화 기법을 적용한 훈련 함수"""
    model = model.to(DEVICE)
    
    # Label Smoothing 사용 여부에 따른 criterion 선택
    if use_label_smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # AdamW optimizer 사용 (더 좋은 weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4, betas=(0.9, 0.999))
    
    # Warmup + Cosine Annealing scheduler
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=5, total_epochs=epochs, 
                                   base_lr=LEARNING_RATE, min_lr=1e-6)
    
    # Early stopping
    best_val_acc = 0
    patience = 20  # 더 긴 patience
    patience_counter = 0
    
    # History tracking
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': [], 'lr': []
    }
    
    print(f"\nTraining {model_name} with Advanced Techniques...")
    print(f"Device: {DEVICE}")
    print(f"Mixup: {use_mixup}, Label Smoothing: {use_label_smoothing}")
    model_info = get_model_info(model)
    print(f"Model Parameters: {model_info['total_params']:,}")
    print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Learning rate scheduling
        current_lr = scheduler.step(epoch)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # Mixup 적용
            if use_mixup and np.random.rand() < 0.5:  # 50% 확률로 mixup 적용
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
                
                optimizer.zero_grad()
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                loss.backward()
                optimizer.step()
                
                # Accuracy 계산 (원본 라벨로)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += (lam * predicted.eq(labels_a).float() + 
                           (1 - lam) * predicted.eq(labels_b).float()).sum().item()
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar):.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc = evaluate_model_detailed(model, val_loader, nn.CrossEntropyLoss())
        
        # Test (for monitoring)
        test_acc = evaluate_model(model, test_loader)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        history['lr'].append(current_lr)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_name.lower()}_advanced_cifar100.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    return history, model_info

def evaluate_model_detailed(model, data_loader, criterion):
    """상세한 모델 평가 - loss와 accuracy 반환"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate_model(model, test_loader):
    """기본 모델 평가 - accuracy만 반환"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def analyze_misclassifications(model, test_loader, num_samples=20):
    """오분류 분석"""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            # 오분류된 샘플 찾기
            mask = predicted != labels
            if mask.any():
                for i in range(len(mask)):
                    if mask[i] and len(misclassified) < num_samples:
                        misclassified.append({
                            'image': images[i].cpu(),
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'confidence': F.softmax(outputs[i], dim=0).max().item()
                        })
            
            if len(misclassified) >= num_samples:
                break
    
    return misclassified

# Dataset classes
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

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        if shuffle:
            from random import shuffle
            shuffle(self.indices)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.idx:self.idx + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.idx += self.batch_size
        return batch
    
# 메인 실험 함수
def run_complete_experiment():
    """과제 요구사항에 맞는 완전한 실험 실행"""
    print("CIFAR-100 Complete Classification Experiment")
    print("=" * 60)
    
    # 데이터셋 준비 (전체 데이터셋 사용)
    print("Preparing full datasets...")
    
    train_dataset_torch = CIFAR100TorchDataset(train_dataset, transform=train_transform)
    test_dataset_torch = CIFAR100TorchDataset(test_dataset, transform=val_test_transform)
    
    # 훈련 데이터를 train/validation으로 분할
    train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset_torch))
    val_size = len(train_dataset_torch) - train_size
    train_subset, val_subset = random_split(train_dataset_torch, [train_size, val_size])
    
    # Validation 데이터에는 augmentation 적용하지 않음
    val_dataset_torch = CIFAR100TorchDataset(train_dataset, transform=val_test_transform)
    val_indices = val_subset.indices
    val_subset_clean = torch.utils.data.Subset(val_dataset_torch, val_indices)
    
    # DataLoader 생성
    train_loader_torch = TorchDataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader_torch = TorchDataLoader(val_subset_clean, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader_torch = TorchDataLoader(test_dataset_torch, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset_torch)}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Max epochs: {EPOCHS}")
    
    # 모델 선택 옵션
    models = {
        'SEResNet18': SEResNet18(),  # SE-ResNet18 (가장 최적화된 모델)
    }
    

    
    results = {}
    
    # 각 모델 훈련
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        # 고급 최적화 기법 사용
        history, model_info = train_model_advanced(
            model, train_loader_torch, val_loader_torch, test_loader_torch, model_name, 
            use_mixup=True, use_label_smoothing=True
        )
        
        # 최종 테스트 성능 평가
        final_test_acc = evaluate_model(model, test_loader_torch)
        
        # 과적합/언더피팅 분석
        best_val_acc = max(history['val_acc'])
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        
        # 성능 요약 출력
        print(f"\n{'='*50}")
        print(f"{model_name} FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Test Accuracy: {final_test_acc:.2f}%")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
        print(f"Parameters: {model_info['total_params']:,}")
        print(f"Model Size: {model_info['model_size_mb']:.2f} MB")
        print(f"Total Epochs: {len(history['train_acc'])}")
        
        # 과적합 분석
        overfitting_gap = final_train_acc - final_val_acc
        if overfitting_gap > 10:
            print(f"과적합 감지: Train-Val gap = {overfitting_gap:.2f}%")
        elif overfitting_gap < 5:
            print(f"적절한 일반화: Train-Val gap = {overfitting_gap:.2f}%")
        else:
            print(f"보통 수준: Train-Val gap = {overfitting_gap:.2f}%")
        
        results[model_name] = {
            'model': model,
            'history': history,
            'model_info': model_info,
            'final_test_acc': final_test_acc,
        }
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"{'='*60}")
    
    return results

# 데이터 로딩 (기본 데이터셋 사용)
print("CIFAR-100 데이터셋 로딩 중...")
train_dataset = Dataset("cifar-100-python/train")
test_dataset = Dataset("cifar-100-python/test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 첫 번째 샘플 확인 (선택적)
if SHOW_SAMPLE_IMAGE:
    print("첫 번째 훈련 샘플 확인 (시각화 비활성화)")
else:
    # 간단한 데이터 확인만
    sample = train_dataset.__getitem__(0)
    print(f"데이터 로딩 확인: {sample['filenames']} (라벨: {sample['fine_labels']})")

# 바로 실험 시작
print("모델 훈련 시작")
print("=" * 50)


results = run_complete_experiment()