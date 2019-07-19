import cv2
vc=cv2.VideoCapture(r"C:\Users\Administrator\Desktop\test_picture\mda-jfaid6sejf0x2kvv.mp4")
c=1
if vc.isOpened():
	rval,frame=vc.read()
else:
	rval=False
while rval:
	rval,frame=vc.read()
	cv2.imwrite(r'C:\Users\Administrator\Desktop\test_picture\ab\ab\ab\1keyframe_'+str(c)+'.jpg',frame)
	c=c+1
	cv2.waitKey(1)
vc.release()
