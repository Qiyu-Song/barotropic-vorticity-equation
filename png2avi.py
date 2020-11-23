import os
import cv2
import time


# 图片合成视频
def picvideo(png_path, size, fps, avi_fn):
    # path: png file path
    filelist = os.listdir(png_path)  # get all files in the path
    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')  # 不同视频编码对应不同视频格式（例：'I','4','2','0' 对应avi格式）

    video = cv2.VideoWriter(avi_fn, fourcc, fps, size)

    for item in filelist:
        if item.endswith('.png'):  # 判断图片后缀是否是.png
            item = png_path + '/' + item
            img = cv2.imread(item)  # 使用opencv读取图像，直接返回numpy.ndarray 对象，通道顺序为BGR ，注意是BGR，通道值默认范围0-255。
            video.write(img)  # 把图片写进视频

    video.release()  # 释放


picvideo('./figure/', (640, 480), 25, './vort.avi')

