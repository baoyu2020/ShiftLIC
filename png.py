import os
import sys

def delete_all_png_files_recursive(start_directory):
    """
    Recursively finds and deletes all .png files within the specified
    start_directory and all its subdirectories.
    Asks for confirmation before deleting ALL found files.

    Args:
        start_directory (str): The full path to the top-level directory to search.
    """
    # 1. Validate the starting directory path
    if not os.path.isdir(start_directory):
        print(f"错误：目录 '{start_directory}' 不存在或不是一个有效的目录。")
        return

    png_files_found = []
    print(f"正在递归扫描目录 '{start_directory}' 及其所有子目录中的 .png 文件...")

    # 2. Use os.walk to traverse the directory tree
    # os.walk yields a 3-tuple for each directory: (dirpath, dirnames, filenames)
    for dirpath, dirnames, filenames in os.walk(start_directory):
        for filename in filenames:
            # 3. Check if the file ends with .png (case-insensitive)
            if filename.lower().endswith('.log'):
                # Construct the full path to the file
                full_path = os.path.join(dirpath, filename)
                png_files_found.append(full_path)

    # 4. Check if any PNG files were found
    if not png_files_found:
        print(f"在 '{start_directory}' 及其子目录中没有找到 .png 文件。")
        return

    # 5. CRITICAL: List all files and ask for confirmation
    print("\n" + "="*60)
    print("警告：以下所有 .png 文件将被从以下位置永久删除:")
    print("请仔细检查列表！")
    print("="*60)
    for file_path in png_files_found:
        print(f"- {file_path}")
    print("="*60)

    # Force user to type 'yes' to confirm
    confirm = input(f"\n你绝对确定要删除所有这 {len(png_files_found)} 个文件吗？\n"
                  f"请输入 'yes' 确认删除，输入其他任何内容取消: ").strip().lower()

    if confirm != 'yes':
        print("操作已取消。")
        return

    # 6. Proceed with deletion
    print("\n开始删除...")
    deleted_count = 0
    error_count = 0
    for file_path in png_files_found:
        try:
            os.remove(file_path)
            # print(f"已删除: {file_path}") # 取消注释以查看每个文件的删除信息，但如果文件很多会很冗长
            deleted_count += 1
        except OSError as e:
            print(f"删除文件时出错 '{file_path}': {e}")
            error_count += 1

    print(f"\n操作完成。")
    print(f"成功删除 {deleted_count} 个 .png 文件。")
    if error_count > 0:
        print(f"删除失败 {error_count} 个文件。")

# --- How to Use ---
if __name__ == "__main__":
    # Prompt the user for the top-level directory path
    # Example Linux path: /home/user/project/data
    # Example Windows path: C:\Users\User\Documents\data
    # top_level_folder = input("请输入要递归删除 PNG 文件的顶级文件夹完整路径 (例如 /path/to/your/data): ").strip()
    top_level_folder = "/data1/home/ynbao/Testcodec/ShiftLIC/data"
    if top_level_folder:
        delete_all_png_files_recursive(top_level_folder)
    else:
        print("未输入路径，操作取消。")