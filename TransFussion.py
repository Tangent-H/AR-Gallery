import cv2
import numpy as np

def transpose3D(obj : np.ndarray, bg, spectiveTransform) -> np.ndarray:
    x1, y1 = spectiveTransform[0]
    x2, y2 = spectiveTransform[1]
    x3, y3 = spectiveTransform[2]
    x4, y4 = spectiveTransform[3]
    x_max = max(x1, x2, x3, x4)
    x_min = min(x1, x2, x3, x4)
    y_max = max(y1, y2, y3, y4)
    y_min = min(y1, y2, y3, y4)
    _k = min(obj.shape[1]/(x_max - x_min), obj.shape[0]/(y_max - y_min))
    _x1 = (x1 - x_min) * _k
    _y1 = (y1 - y_min) * _k
    _x2 = (x2 - x_min) * _k
    _y2 = (y2 - y_min) * _k
    _x3 = (x3 - x_min) * _k
    _y3 = (y3 - y_min) * _k
    _x4 = (x4 - x_min) * _k
    _y4 = (y4 - y_min) * _k
    center_bg = [(x1 + x2 + x3 + x4) // 4, (y1 + y2 + y3 + y4) // 4]
    dst_points = np.float32([[_x1, _y1], [_x2, _y2], [_x3, _y3], [_x4, _y4]])
    src_points = np.float32([[0, 0], [obj.shape[1], 0], [obj.shape[1], obj.shape[0]], [0, obj.shape[0]]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(obj, M, (obj.shape[1], obj.shape[0]))
    return warped_image, center_bg, M, 1/_k

def main_color_detect(image) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    values = hsv[:,:,2]
    hist = cv2.calcHist([values], [0], None, [256], [0, 256])
    color_idx = np.argmax(hist)
    color_bgr = np.array([[[color_idx, color_idx, color_idx]]])
    # print(color_bgr)
    return color_bgr

def crop_image_outside_bounds(image, center, shape_bg):
    # 计算图像边界框的坐标
    center_x = center[0]
    center_y = center[1]
    max_height = shape_bg[0]
    max_width = shape_bg[1]
    x1 = int(center_x - image.shape[1] / 2)
    y1 = int(center_y - image.shape[0] / 2)
    x2 = int(center_x + image.shape[1] / 2)
    y2 = int(center_y + image.shape[0] / 2)
    
    # 初始化裁剪区域
    crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, max_width, max_height
    
    cut = 0
    if x1 < 0:  # 图像左侧超出边界
        crop_x1 = -x1
        cut += 2
    if y1 < 0:  # 图像顶部超出边界
        crop_y1 = -y1
        cut += 20
    if x2 > max_width:  # 图像右侧超出边界
        crop_x2 = max_width
        cut += 3
    if y2 > max_height:  # 图像底部超出边界
        crop_y2 = max_height
        cut += 30

    if cut == 0:
        return image, center
        
    # 裁剪图像
    cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
    
    if cut%10 == 2:
        center[0] = cropped_image.shape[1] // 2 
    elif cut%10 == 3:
        center[0] = max_width - cropped_image.shape[1] // 2
    elif cut%10 == 5:
        center[0] = max_width // 2
    if cut//10 == 2:
        center[1] = cropped_image.shape[0] // 2
    elif cut//10 == 3:
        center[1] = max_height - cropped_image.shape[0] // 2
    elif cut//10 == 5:
        center[1] = max_height // 2
    # cv2.imshow('cropped_image', cropped_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_image, center

def fussion(background : np.ndarray, obj : np.ndarray, center, k) -> np.ndarray:
    width_obj, height_obj, channels_ = obj.shape
    obj_rs = cv2.resize(obj, (int(width_obj*k), int(height_obj*k)), interpolation=cv2.INTER_LINEAR)
    # print('[debug]', obj_rs.shape, background.shape, center)
    obj_croped, center = crop_image_outside_bounds(obj_rs, center, background.shape) # 
    # print('[debug]', obj_croped.shape, background.shape, center)
    mask = 255 * np.ones(obj_croped.shape, obj_croped.dtype)
    mixed_clone = cv2.seamlessClone(obj_croped, background, mask, center, cv2.NORMAL_CLONE)
    return mixed_clone

def TransFussion(bg, obj, spectiveTransform, k):
    """
    inputs: bg:背景图, obj:物体原图, spectiveTransform:角点元组, k:缩放比例
    output: 融合之后的图像

    提示:当k值过大, 融合后的物体图超过背景图范围时, 程序会报错。
    """
    if isinstance(bg, str):
        bg = cv2.imread(bg)
    if isinstance(obj, str):
        obj = cv2.imread(obj)

    warped_image, center_bg, Mask, _k = transpose3D(obj, bg, spectiveTransform)

    color_bgr = main_color_detect(obj)
    mask = cv2.warpPerspective(np.ones(obj.shape[:2], dtype=np.uint8), Mask, dsize=(obj.shape[1], obj.shape[0]))
    bool_mask = mask > 0
    warped_image[~bool_mask] = color_bgr

    mixed_image = fussion(bg, warped_image, center_bg, k*_k)
    return mixed_image

if __name__ == '__main__':
    # 读取原始图片
    image = cv2.imread('../figures/example.jpg')
    bg = cv2.imread('../figures/example2.jpg')

    mixed_image = TransFussion(bg, image, [[70, 60], [200, 80], [250, 200], [30, 220]], 0.5)

    # cv2.imshow('mixed_clone', mixed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()