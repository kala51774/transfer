import os


def create_directories(epoch, current_dir='.'):
    # 定义基本文件夹名称
    base_folder = f'epoch{epoch}'
    base_path = os.path.join(current_dir, base_folder)

    # 定义子文件夹名称
    real_folder = os.path.join(base_path, 'real')
    fake_folder = os.path.join(base_path, 'fake')
    compare_folder = os.path.join(base_path, 'compare')

    # 检查并创建基本文件夹
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created directory: {base_path}")

    # 检查并创建real子目录
    if not os.path.exists(real_folder):
        os.makedirs(real_folder)
        print(f"Created directory: {real_folder}")
    else:
        print(f"Directory already exists: {real_folder}")

    # 检查并创建fake子目录
    if not os.path.exists(fake_folder):
        os.makedirs(fake_folder)
        print(f"Created directory: {fake_folder}")
    else:
        print(f"Directory already exists: {fake_folder}")

    # 检查并创建compare子目录
    if not os.path.exists(compare_folder):
        os.makedirs(compare_folder)
        print(f"Created directory: {compare_folder}")
    else:
        print(f"Directory already exists: {compare_folder}")


# 使用示例
create_directories(2, current_dir='results/hayao')