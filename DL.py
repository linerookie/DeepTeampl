import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import time
import os
import json
from collections import defaultdict
import seaborn as sns

print("CIFAR-100 Full Training Mode")
print("=" * 50)
print("훈련 데이터: 50,000개 ")
print("테스트 데이터: 10,000개") 
print("에포크: 100")
print("배치 크기: 64")
print("Early stopping patience: 15")
print("=" * 50)

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
    print(f"✅ CIFAR-100 클래스 이름 로드 완료: {len(CIFAR100_CLASSES)}개 클래스")
except:
    CIFAR100_CLASSES = [f"Class_{i}" for i in range(100)]  # 간단한 대체
    print("⚠️  클래스 이름 대신 번호 사용: Class_0, Class_1, ...")

# Baseline Simple CNN Model (요구사항: 직접 설계한 CNN 모델)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=100):
        super(SimpleCNN, self).__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Conv Block 2
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Conv Block 3
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Conv Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = self.fc3(x)
        
        return x

# ResNet Implementation (사전학습 모델 기반)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# EfficientNet Implementation
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expand_ratio != 1 else None
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expand_ratio != 1 else None
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                                       stride, padding=kernel_size//2, groups=expanded_channels, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze and Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se_reduce = nn.Conv2d(expanded_channels, se_channels, 1)
        self.se_expand = nn.Conv2d(se_channels, expanded_channels, 1)
        
        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        identity = x
        
        # Expansion
        if self.expand_conv is not None:
            x = F.relu6(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze and Excitation
        se = F.adaptive_avg_pool2d(x, 1)
        se = F.relu(self.se_reduce(se))
        se = torch.sigmoid(self.se_expand(se))
        x = x * se
        
        # Output
        x = self.project_bn(self.project_conv(x))
        
        # Skip connection
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + identity
            
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, 1),
            MBConvBlock(16, 24, 3, 2, 6),
            MBConvBlock(24, 24, 3, 1, 6),
            MBConvBlock(24, 40, 5, 2, 6),
            MBConvBlock(40, 40, 5, 1, 6),
            MBConvBlock(40, 80, 3, 2, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 192, 5, 2, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 320, 3, 1, 6),
        )
        
        # Final conv
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Classifier
        self.classifier = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.blocks(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    transforms.RandomErasing(p=0.1)
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

def train_model_enhanced(model, train_loader, val_loader, test_loader, model_name, epochs=EPOCHS):
    """향상된 훈련 함수 - validation 포함"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Early stopping
    best_val_acc = 0
    patience = 15  # 원래대로
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

def plot_misclassifications(misclassified, model_name, use_class_names=True):
    """오분류 시각화"""
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Misclassified Examples', fontsize=16)
    
    for i, sample in enumerate(misclassified[:20]):
        row, col = i // 5, i % 5
        
        # 이미지 정규화 해제
        image = sample['image']
        mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        axes[row, col].imshow(image.permute(1, 2, 0))
        
        if use_class_names and len(CIFAR100_CLASSES) == 100:
            # 클래스 이름 사용
            true_label = CIFAR100_CLASSES[sample["true_label"]]
            pred_label = CIFAR100_CLASSES[sample["predicted_label"]]
        else:
            # 클래스 번호 사용 (더 간단)
            true_label = f"Class {sample['true_label']}"
            pred_label = f"Class {sample['predicted_label']}"
        
        axes[row, col].set_title(
            f'True: {true_label}\n'
            f'Pred: {pred_label}\n'
            f'Conf: {sample["confidence"]:.3f}', 
            fontsize=8
        )
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_misclassifications.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comprehensive_results(results_dict):
    """종합적인 결과 시각화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    model_names = list(results_dict.keys())
    
    # Training & Validation Loss
    axes[0, 0].set_title('Training & Validation Loss')
    for model_name in model_names:
        history = results_dict[model_name]['history']
        axes[0, 0].plot(history['train_loss'], label=f'{model_name} Train', linestyle='-')
        axes[0, 0].plot(history['val_loss'], label=f'{model_name} Val', linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Training & Validation Accuracy
    axes[0, 1].set_title('Training & Validation Accuracy')
    for model_name in model_names:
        history = results_dict[model_name]['history']
        axes[0, 1].plot(history['train_acc'], label=f'{model_name} Train', linestyle='-')
        axes[0, 1].plot(history['val_acc'], label=f'{model_name} Val', linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Test Accuracy Comparison
    test_accs = [results_dict[name]['final_test_acc'] for name in model_names]
    colors = ['blue', 'red', 'green']
    axes[0, 2].bar(model_names, test_accs, color=colors[:len(model_names)], alpha=0.7)
    axes[0, 2].set_title('Final Test Accuracy Comparison')
    axes[0, 2].set_ylabel('Accuracy (%)')
    for i, acc in enumerate(test_accs):
        axes[0, 2].text(i, acc + 0.5, f'{acc:.2f}%', ha='center', va='bottom')
    
    # Model Parameters Comparison
    param_counts = [results_dict[name]['model_info']['total_params'] for name in model_names]
    axes[1, 0].bar(model_names, param_counts, color=colors[:len(model_names)], alpha=0.7)
    axes[1, 0].set_title('Model Parameters Comparison')
    axes[1, 0].set_ylabel('Number of Parameters')
    for i, count in enumerate(param_counts):
        axes[1, 0].text(i, count + max(param_counts)*0.01, f'{count:,}', ha='center', va='bottom', rotation=45)
    
    # Model Size Comparison
    model_sizes = [results_dict[name]['model_info']['model_size_mb'] for name in model_names]
    axes[1, 1].bar(model_names, model_sizes, color=colors[:len(model_names)], alpha=0.7)
    axes[1, 1].set_title('Model Size Comparison')
    axes[1, 1].set_ylabel('Size (MB)')
    for i, size in enumerate(model_sizes):
        axes[1, 1].text(i, size + max(model_sizes)*0.01, f'{size:.2f}MB', ha='center', va='bottom')
    
    # Efficiency Score (Accuracy per Parameter)
    efficiency_scores = [test_accs[i] / (param_counts[i] / 1e6) for i in range(len(model_names))]
    axes[1, 2].bar(model_names, efficiency_scores, color=colors[:len(model_names)], alpha=0.7)
    axes[1, 2].set_title('Efficiency Score (Accuracy/Million Parameters)')
    axes[1, 2].set_ylabel('Efficiency Score')
    for i, score in enumerate(efficiency_scores):
        axes[1, 2].text(i, score + max(efficiency_scores)*0.01, f'{score:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_json(results_dict, filename='experiment_results.json'):
    """실험 결과를 JSON 파일로 저장"""
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for model_name, results in results_dict.items():
        json_results[model_name] = {
            'final_test_acc': results['final_test_acc'],
            'model_info': results['model_info'],
            'best_val_acc': max(results['history']['val_acc']),
            'final_train_acc': results['history']['train_acc'][-1],
            'total_epochs': len(results['history']['train_acc'])
        }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {filename}")

# Original dataset classes
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
    
def print_data(data):
    print(data['filenames'])
    print(data['fine_labels'])
    print(data['coarse_labels'])
    img = data['data']
    img_trans = np.transpose(img, (1, 2, 0))
    plt.imshow(img_trans)
    plt.axis('off')
    plt.show()

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
    
    # 모델 선택 옵션 ()
    models = {
        'ResNet18': ResNet18(),
        # 'SimpleCNN': SimpleCNN(),        
        # 'EfficientNet': EfficientNet()   
    }
    

    
    results = {}
    
    # 각 모델 훈련
    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        
        history, model_info = train_model_enhanced(
            model, train_loader_torch, val_loader_torch, test_loader_torch, model_name
        )
        
        # 최종 테스트 성능 평가
        final_test_acc = evaluate_model(model, test_loader_torch)
        
        # 오분류 분석
        print(f"\nAnalyzing misclassifications for {model_name}...")
        misclassified = analyze_misclassifications(model, test_loader_torch)
        if SHOW_MISCLASSIFICATIONS:
            plot_misclassifications(misclassified, model_name)
        
        results[model_name] = {
            'model': model,
            'history': history,
            'model_info': model_info,
            'final_test_acc': final_test_acc,
            'misclassified': misclassified
        }
        
        print(f"{model_name} Final Results:")
        print(f"  - Test Accuracy: {final_test_acc:.2f}%")
        print(f"  - Best Validation Accuracy: {max(history['val_acc']):.2f}%")
        print(f"  - Parameters: {model_info['total_params']:,}")
        print(f"  - Model Size: {model_info['model_size_mb']:.2f} MB")
    
    # 종합 결과 시각화
    print(f"\n{'='*60}")
    print("Generating comprehensive comparison...")
    if SHOW_RESULT_PLOTS:
        plot_comprehensive_results(results)
    
    # 결과 저장
    save_results_to_json(results)
    
    # 최종 모델 저장
    for model_name, result in results.items():
        torch.save(result['model'].state_dict(), f'final_{model_name.lower()}_cifar100.pth')
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETED!")
    print("="*60)
    print("\nFinal Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_test_acc'], reverse=True)
    for i, (model_name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {model_name}: {result['final_test_acc']:.2f}% "
              f"(Params: {result['model_info']['total_params']:,})")
    
    return results

# 데이터 로딩 (기본 데이터셋 사용)
print("📁 CIFAR-100 데이터셋 로딩 중...")
train_dataset = Dataset("cifar-100-python/train")
test_dataset = Dataset("cifar-100-python/test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 첫 번째 샘플 확인 (선택적)
if SHOW_SAMPLE_IMAGE:
    print("🖼️ 첫 번째 훈련 샘플:")
    print_data(train_dataset.__getitem__(0))
else:
    # 간단한 데이터 확인만
    sample = train_dataset.__getitem__(0)
    print(f"✅ 데이터 로딩 확인: {sample['filenames']} (라벨: {sample['fine_labels']})")

# 바로 실험 시작
print("모델 훈련 시작")
print("=" * 50)


results = run_complete_experiment()