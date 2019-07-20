import cv2
import os
def save_img():
    video_path = r'C:\Users\Administrator\Desktop\test/'
    videos = os.listdir(video_path)
    for video_name in videos:
        file_name = video_name.split('.')[0]
        folder_name =  file_name
        os.makedirs(folder_name,exist_ok=True)
        vc = cv2.VideoCapture(video_path+video_name) #读入视频文件
        c = 1
        if vc.isOpened():  # 判断是否正常打开
            rval, frame = vc.read()
        else:
            rval = False
 
        timeF = 2  # 视频帧计数间隔频率
 
        while rval:  # 循环读取视频帧
            rval, frame = vc.read()
            pic_path = folder_name + '/'
            if (c % timeF == 0):  # 每隔timeF帧进行存储操作
                cv2.imwrite(r'C:\Users\Administrator\Desktop\test\test1212/'+ file_name + '_' + str(c) + '.jpg', frame)  # 存储为图像,保存名为 文件夹名_数字（第几个文件）.jpg
            c = c + 1
            cv2.waitKey(1)
        vc.release()
save_img()
