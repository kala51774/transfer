import sys
import os
import shutil


def rename_and_copy_files(src_dir_a, src_dir_b, dest_dir):
    # 确保目标目录存在
    os.makedirs(dest_dir, exist_ok=True)

    # 文件计数器
    file_counter = 0

    # 定义文件复制和重命名的函数
    def copy_and_rename_files(source_dir):
        nonlocal file_counter
        for filename in os.listdir(source_dir):
            file_path = os.path.join(source_dir, filename)
            # 确保是文件而不是目录
            if os.path.isfile(file_path):
                # 构造新文件名和路径
                new_filename = f"{file_counter}.jpg"
                new_file_path = os.path.join(dest_dir, new_filename)
                # 复制并重命名文件
                shutil.copy(file_path, new_file_path)
                # 更新文件计数器
                file_counter += 1

    # 复制并重命名两个源目录中的文件
    copy_and_rename_files(src_dir_a)
    copy_and_rename_files(src_dir_b)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <src_dir_a> <src_dir_b> <dest_dir>")
        sys.exit(1)

    src_dir_a = sys.argv[1]
    src_dir_b = sys.argv[2]
    dest_dir = sys.argv[3]

    rename_and_copy_files(src_dir_a, src_dir_b, dest_dir)