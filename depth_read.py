

import cv2
import numpy as np
if __name__ == "__main__":
    depth_img = cv2.imread(r"C:\Users\23113\Desktop\Depth_Photo.png", cv2.IMREAD_UNCHANGED)
    if depth_img.dtype != 'uint8':
        # 将深度图像归一化到0-255范围
        depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        # 转换为8位图像
        depth_img = np.uint8(depth_img_normalized)
    color_depth_map = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)

    cv2.imshow('depth_image.png', color_depth_map)
    depth_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    
    cv2.imshow("Depth Image", depth_img)
    cv2.imshow("Color Mapped Depth Image", depth_color)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
