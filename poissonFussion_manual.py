from PIL import Image
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time

# 图像读取
def read_image(image_path):
    return np.array(Image.open(image_path))

# 图像准备
def prepare_images(foreground, background, mask):
    background = Image.open(background)
    background = np.array(background, dtype=np.float32)
    foreground = Image.open(foreground).resize(mask[1])
    foreground = np.array(foreground, dtype=np.float32)
    return foreground, background

def _cal_gradient2(image, background):
    height, width = image.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                output[i, j] = background[i, j]
            else:
                output[i, j] = -4 * image[i, j] + (image[i-1, j] + image[i+1, j] + image[i, j-1] + image[i, j+1])
    return output

def cal_A(shape):
    width, height = shape
    length = height*width
    A = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        up = i - width
        down = i + width
        left = i - 1
        right = i + 1
        row = i // width
        col = i % width
        # 判断边缘点
        if row == 0 or row == height-1 or col == 0 or col == width-1:
            A[i, i] = 1
        else:
            A[i, i] = -4
            A[i, up] = 1
            A[i, down] = 1
            A[i, left] = 1
            A[i, right] = 1
    return A


# 建立泊松方程
def poisson_equation(foreground, background, mask):
    print(mask[0, 0], mask[0, 1], mask[1, 0], mask[1, 1])
    background_cut = background[mask[0, 0]:mask[0, 0]+mask[1, 1], mask[0, 1]:mask[0, 1]+mask[1, 0]]
    delta2_B = _cal_gradient2(foreground, background_cut)
    delta2_B_vector = delta2_B.reshape(delta2_B.shape[0] * delta2_B.shape[1], 3)
    A = cal_A(mask[-1])
    x = np.zeros((mask[1, 1]*mask[1, 0], 3), dtype=np.float32)
    A_sparse = csc_matrix(A)
    for i in range(3):
        x_solution = spsolve(A_sparse, delta2_B_vector[:, i])
        x[:, i] = np.array(x_solution)
    x = x.reshape(mask[1, 1], mask[1, 0], 3)
    x = np.clip(x, 0, 255)
    return x

# 图像融合
def blend_images(foreground_mixed, background, mask):
    mixed_pic = np.copy(background)
    mixed_pic[mask[0, 0]:mask[0, 0]+mask[1, 1], mask[0, 1]:mask[0, 1]+mask[1, 0]] = foreground_mixed
    return mixed_pic.astype(np.uint8)

# 主函数
def main(foreground_path, background_path):
    mask = np.array([[400, 400], [300, 200]]) # 中心位置 边框大小
    foreground, background = prepare_images(foreground_path, background_path, mask)
    # start = time.time()
    foreground_mixed = poisson_equation(foreground, background, mask)
    # end = time.time()
    # print('Time:', end - start)
    foreground_mixed = (foreground_mixed)
    print(foreground_mixed[0, 0, 0].dtype)
    output_image = blend_images(foreground_mixed, background, mask)
    
    # 保存或显示结果
    Image.fromarray(output_image).save('./test/fussion_output.jpg')
    # Image.fromarray(output_image).show()

if __name__ == "__main__":
    main('./test/fussion_plane.jpg', './test/fussion_underwater.jpg')