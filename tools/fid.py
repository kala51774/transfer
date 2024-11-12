import os
import torch
from sympy.matrices.expressions.slice import normalize
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

def load_and_preprocess_images(folder):
    images = []
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
    return torch.stack(images)

def calculate_fid_and_save_result(input_dir):
    # 定义子文件夹路径
    real_folder = os.path.join(input_dir, 'real')
    fake_folder = os.path.join(input_dir, 'fake')
    result_file = os.path.join(input_dir, 'result')

    # 加载并预处理图像
    real_images = load_and_preprocess_images(real_folder)
    fake_images = load_and_preprocess_images(fake_folder)

    # 确保图像数量相同
    min_length = min(len(real_images), len(fake_images))
    real_images = real_images[:min_length]
    fake_images = fake_images[:min_length]
    # 初始化FID计算对象
    fid = FrechetInceptionDistance(feature=64,normalize=True)
    d_type = torch.float32
    fid.set_dtype(d_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device ='cpu'
    fid.to(device)
    real_images=real_images.to(dtype=d_type).to(device)
    fake_images=fake_images.to(dtype=d_type).to(device)

    print(real_images.dtype, fake_images.dtype)
    # 更新FID计算对象
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    # 计算FID
    fid_score = fid.compute()

    # 保存结果到文件
    with open(result_file, 'w') as f:
        f.write(f"FID score: {fid_score}")

    # print(f"FID score: {fid_score}")
    return fid_score

# 使用示例
input_dir = 'E:\\Paper\\paper_with_code\\DEMAL_Model-main\\results\\hayao\\epoch_0'
fid_value = calculate_fid_and_save_result(input_dir)
print(f"Returned FID value: {fid_value}")