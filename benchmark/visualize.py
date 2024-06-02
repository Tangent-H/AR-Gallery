import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_bounding_box(image, correct):
    color = (0, 255, 0) if correct else (255, 0, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert color for Matplotlib
    color = tuple(c / 255.0 for c in color)  # Normalize color for Matplotlib
    return image, color

def create_combined_image(xs, cv_flags, ours_flags, output_file='combined_result.png'):
    fig, axs = plt.subplots(len(xs), 2, figsize=(10, 3 * len(xs)))
    
    for i, x in enumerate(xs):
        cv_image = cv2.imread(f'results/cv_{x}.jpg')
        ours_image = cv2.imread(f'results/ours_{x}.jpg')
        
        if cv_image is None or ours_image is None:
            print(f'Error: Could not read images cv_{x}.jpg or ours_{x}.jpg')
            continue
        
        cv_image, cv_color = draw_bounding_box(cv_image, cv_flags[i])
        ours_image, ours_color = draw_bounding_box(ours_image, ours_flags[i])
        
        axs[i, 0].imshow(cv_image)
        if i == 0:
            axs[i, 0].set_title('OpenCV Method', fontsize=20)
        axs[i, 0].axis('off')
        axs[i, 0].add_patch(plt.Rectangle((0, 0), cv_image.shape[1], cv_image.shape[0], edgecolor=cv_color, linewidth=3, facecolor='none'))
        
        axs[i, 1].imshow(ours_image)
        if i == 0:
            axs[i, 1].set_title('Our Method', fontsize=20)
        axs[i, 1].axis('off')
        axs[i, 1].add_patch(plt.Rectangle((0, 0), ours_image.shape[1], ours_image.shape[0], edgecolor=ours_color, linewidth=3, facecolor='none'))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Adjust space between rows
    plt.savefig(output_file, bbox_inches='tight')
    plt.show()

# 示例调用
xs = [121, 380, 280, 327,279,164]  # 替换为实际的文件名索引
cv_flags = [0, 0, 0,0,0,1]  # 替换为实际的OpenCV方法识别结果
ours_flags = [1, 1, 1,1,0,0]  # 替换为实际的我们的方法识别结果

create_combined_image(xs, cv_flags, ours_flags)
