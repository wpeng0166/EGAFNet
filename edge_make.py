import os
import cv2
import numpy as np


def imread_unicode(path, flags=cv2.IMREAD_GRAYSCALE):
    """支持中文路径的图像读取函数"""
    try:
        with open(path, 'rb') as f:
            img_array = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(img_array, flags)
            return img
    except Exception as e:
        print(f"⚠️ 读取失败: {path}, 错误信息: {e}")
        return None

def imwrite_unicode(path, img):
    """支持中文路径的图像保存函数"""
    try:
        is_success, buffer = cv2.imencode('.png', img)

        if not is_success:
            print(f"❌ 图像编码失败: {path}")
            return False

        with open(path, 'wb') as f:
            f.write(buffer)
        return True
    except Exception as e:
        print(f"⚠️ 保存失败: {path}, 错误信息: {e}")
        return False


def calculate_edge(datapath):
    # subsets = ['train', 'val', 'test']
    subsets = ['train', 'test']
    for subset in subsets:
        label_dir = os.path.join(datapath, subset, 'label')
        edge_dir = os.path.join(datapath, subset, 'edge')

        if not os.path.exists(label_dir):
            print(f"ℹ️ 目录不存在，跳过: {label_dir}")
            continue

        if not os.path.exists(edge_dir):
            os.makedirs(edge_dir)
            print(f"📁 已创建目录: {edge_dir}")

        filenames = os.listdir(label_dir)
        if not filenames:
            print(f"ℹ️ 目录为空，跳过: {label_dir}")
            continue

        print(f"\n--- 正在处理子集: {subset} ---")
        for filename in filenames:
            # 确保处理的是文件而不是子目录
            file_path = os.path.join(label_dir, filename)
            if not os.path.isfile(file_path):
                continue

            save_path = os.path.join(edge_dir, filename)

            if os.path.getsize(file_path) == 0:
                print(f"❌ 文件为空: {file_path}")
                continue

            image = imread_unicode(file_path, flags=cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            # 使用 Sobel 边缘检测
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)


            abs_sobel_x = np.absolute(sobel_x)
            abs_sobel_y = np.absolute(sobel_y)
            sobel_mag = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)

            sobel_mag = cv2.convertScaleAbs(sobel_mag)
            _, binary_edge = cv2.threshold(sobel_mag, 20, 255, cv2.THRESH_BINARY)

            if imwrite_unicode(save_path, binary_edge):
                print(f"✅ 已保存边缘图: {save_path}")
            else:
                # imwrite_unicode 内部已经打印了详细错误
                print(f"❌ 未能保存边缘图: {save_path}")


if __name__ == '__main__':
    datapath = r'/home/s_wp/workspace/data/LEVIR-CD256full/'
    if not os.path.exists(datapath):
        print(f"‼️‼️‼️ 根目录不存在: {datapath}")
    else:
        calculate_edge(datapath)