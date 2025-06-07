import torch
import torch.nn as nn # nn.Linear 등을 위해 필요
import torchvision.transforms as transforms
from PIL import Image # 이미지 로딩을 위해 필요
import os
import time

# --- 이전 팀원 코드에서 가져온 모델 정의 (예시: ResNet18) ---
# 실제 72%를 달성한 모델의 정의와 동일해야 합니다.
# 이 부분은 팀원분의 스크립트에서 복사해오거나 임포트해야 합니다.
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
        out = torch.relu(self.bn1(self.conv1(x))) # F.relu 대신 torch.relu 사용 가능
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
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
        for stride_val in strides: # stride 변수명 충돌 피하기 위해 변경
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4) # F.avg_pool2d 대신 nn.functional.avg_pool2d
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes=100): # num_classes 인자 추가
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes) # ResNet 클래스에 전달
# --- 모델 정의 끝 ---

# 추론 결과를 생성하는 함수
def generate_competition_submission(model_path, model_architecture_fn, cimages_dir, output_file_path, device):
    """
    학습된 모델로 CImages에 대한 추론을 수행하고 결과 파일을 생성합니다.

    Args:
        model_path (str): 학습된 모델 가중치 파일 (.pth) 경로.
        model_architecture_fn (function): 모델 구조를 반환하는 함수 (예: ResNet18).
        cimages_dir (str): CImages 폴더 경로 (예: './Dataset/CImages/').
        output_file_path (str): 생성될 결과 파일 경로 (예: './result_팀명_날짜_시간.txt').
        device (torch.device): 추론을 수행할 장치 ('cuda' 또는 'cpu').
    """
    print(f"모델 로딩 중: {model_path}")
    # 1. 모델 구조 인스턴스화 및 가중치 로드
    model = model_architecture_fn(num_classes=100) # CIFAR-100은 100개 클래스
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"모델 가중치 로딩 실패: {e}")
        print("모델 아키텍처와 저장된 가중치가 일치하는지, 파일 경로가 정확한지 확인하세요.")
        return
        
    model.to(device)
    model.eval() # 평가 모드로 설정
    print("모델 로딩 및 평가 모드 설정 완료.")

    # 2. CImages에 사용할 이미지 변환 (val_test_transform과 동일하게)
    # 팀원분의 코드에서 val_test_transform을 가져오거나 동일하게 정의합니다.
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)
    image_transform = transforms.Compose([
        transforms.ToTensor(), # PIL Image를 Tensor로 변환 (0-1 범위)
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])
    print("이미지 변환 설정 완료.")

    results = []
    num_images = 3000 # 경진대회 이미지 수

    print(f"추론 시작: {cimages_dir}의 이미지 처리 중...")
    for i in range(1, num_images + 1):
        # 파일명 형식: 0001.jpg, 0002.jpg, ...
        # 사용자님은 1.jpg ~ 3000.jpg로 언급하셨지만, 경진대회 규정상 000X.jpg 형태일 가능성이 높습니다.
        # 여기서는 000X.jpg 형태로 가정합니다.
        filename_numeric_part = f"{i}" # 1 -> "0001", 10 -> "0010", 1234 -> "1234"
        image_filename = f"{filename_numeric_part}.jpg"
        image_path = os.path.join(cimages_dir, image_filename)

        if not os.path.exists(image_path):
            print(f"경고: 파일을 찾을 수 없습니다 - {image_path}")
            # 파일이 없으면 결과에 누락시키거나, 특정 값으로 채울 수 있습니다.
            # 여기서는 누락된 파일에 대한 결과는 기록하지 않도록 할 수 있지만,
            # 경진대회 규정상 모든 번호에 대한 결과가 필요할 수 있으므로 주의해야 합니다.
            # 우선 빈 레이블(-1)로 기록하거나, 에러를 발생시킬 수 있습니다.
            # results.append(f"{filename_numeric_part},-1") # 예: 오류 또는 누락 표시
            continue


        try:
            # 이미지 로드 및 전처리
            image = Image.open(image_path).convert('RGB') # PIL Image로 로드
            image_tensor = image_transform(image).unsqueeze(0) # 배치 차원 추가
            image_tensor = image_tensor.to(device)

            # 추론
            with torch.no_grad(): # 기울기 계산 비활성화
                outputs = model(image_tensor)
                _, predicted_label = torch.max(outputs.data, 1)
            
            results.append(f"{filename_numeric_part},{predicted_label.item()}")

            if i % 300 == 0: # 300개 이미지 처리마다 진행 상황 출력
                print(f"  {i}/{num_images} 이미지 처리 완료...")

        except Exception as e:
            print(f"오류 발생 ({image_filename}): {e}")
            results.append(f"{filename_numeric_part},-1") # 오류 발생 시 -1 레이블 (예시)
            
    print("추론 완료.")

    # 4. 결과 파일 저장
    try:
        with open(output_file_path, 'w') as f:
            f.write("number,label\n") # 헤더 추가
            for line in results:
                f.write(line + "\n")
        print(f"결과 파일 저장 완료: {output_file_path}")
    except Exception as e:
        print(f"결과 파일 저장 중 오류: {e}")


if __name__ == '__main__':
    # --- 사용 예시 ---
    DEVICE_FOR_INFERENCE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 학습된 모델 가중치 파일 경로
    # 예시: 팀원분이 72% 달성한 모델의 .pth 파일 경로
    # 또는 우리가 학습시킨 모델의 경로 (예: './saved_models_v5/EfficientNet-B0_best_val_acc_....pth')
    trained_model_path = './best_ResNet18_cifar100.pth' # 여기에 실제 모델 경로 입력!

    # 2. CImages 폴더 경로
    cimages_folder_path = './Dataset/CImages/'

    # 3. 결과 파일명 (경진대회 규격에 맞게 팀명, 날짜, 시간 포함)
    # 예시: result_가반1조_0602_1000.txt
    team_name_example = "MyTeam" # 실제 팀명으로 변경
    submission_timestamp = time.strftime("%m%d_%H%M") # 예: 0602_1000
    output_filename = f"./result_{team_name_example}_{submission_timestamp}.txt"

    # 4. 모델 아키텍처를 반환하는 함수
    # 만약 팀원분의 모델이 ResNet18이라면 ResNet18 함수를 전달합니다.
    # 우리가 학습시킨 EfficientNet-B0를 사용한다면, EfficientNet-B0를 로드하고 수정하는 함수를 만들어 전달해야 합니다.
    # 여기서는 팀원분의 ResNet18을 예시로 사용합니다.
    model_function = ResNet18 

    print("경진대회 제출 파일 생성 시작...")
    print(f"  사용 모델 가중치: {trained_model_path}")
    print(f"  CImages 경로: {cimages_folder_path}")
    print(f"  출력 파일명: {output_filename}")
    print(f"  추론 장치: {DEVICE_FOR_INFERENCE}")

    if not os.path.exists(trained_model_path):
        print(f"오류: 모델 가중치 파일을 찾을 수 없습니다 - {trained_model_path}")
        print("trained_model_path 변수를 올바른 경로로 수정해주세요.")
    elif not os.path.isdir(cimages_folder_path):
        print(f"오류: CImages 폴더를 찾을 수 없습니다 - {cimages_folder_path}")
        print("cimages_folder_path 변수를 올바른 경로로 수정해주세요.")
    else:
        generate_competition_submission(
            model_path=trained_model_path,
            model_architecture_fn=model_function,
            cimages_dir=cimages_folder_path,
            output_file_path=output_filename,
            device=DEVICE_FOR_INFERENCE
        )
    print("작업 완료.")