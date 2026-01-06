import os
import uuid


def rename_to_numbers_force(folder, ext=".png", start=1):
    """
    强制将文件夹内的文件重命名为 1.png, 2.png...
    通过中间临时文件名解决 "FileExistsError" 冲突。
    """
    if not os.path.exists(folder):
        print(f"❌ 错误: 文件夹不存在 - {folder}")
        return

    # 1. 获取所有文件 (过滤掉隐藏文件和子文件夹)
    # 使用 sort 保证每次运行顺序相对固定（按原文件名字母序）
    files = sorted([f for f in os.listdir(folder) if not f.startswith('.')])

    # 筛选出只是文件的路径
    valid_files = []
    for f in files:
        full_path = os.path.join(folder, f)
        if os.path.isfile(full_path):
            valid_files.append(full_path)

    if not valid_files:
        print("📂 文件夹为空或没有文件。")
        return

    print(f"检测到 {len(valid_files)} 个文件，准备处理...")

    # --- 阶段 1: 全部重命名为临时乱码 ---
    # 这一步是为了腾出 1.png, 2.png 等名字，防止冲突
    temp_paths = []
    for old_path in valid_files:
        # 生成一个唯一的临时名字，例如: temp_uuidxxxx.png
        temp_name = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(folder, temp_name)

        os.rename(old_path, temp_path)
        temp_paths.append(temp_path)

    # --- 阶段 2: 重命名为目标数字 ---
    # 现在文件夹里全是临时文件，可以放心改成 1, 2, 3...
    count = 0
    for i, temp_path in enumerate(temp_paths):
        idx = start + i
        new_name = f"{idx}{ext}"
        new_path = os.path.join(folder, new_name)

        os.rename(temp_path, new_path)
        count += 1

    print(f"✅ 成功! 已将 {count} 个文件重命名为 {start}{ext} 到 {start + count - 1}{ext}")


if __name__ == "__main__":
    rename_to_numbers_force(
        folder=r"/home/wangzhe/2026/IROS/Dataset/Img/apple2",
        ext=".png",  # 统一改成 png
        start=1  # 编号从 1 开始
    )