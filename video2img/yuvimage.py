import cv2
import os
import argparse
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

def yuv2rgb(pathname,filename,W, H, startframe, totalframe, show=False, out=False):
    """
    read yuv file and convert it to a numpy array
    

    Parameters
    ----------
    yuvfilename : TYPE
        文件名.
    W : TYPE
        width.
    H : TYPE
        height.
    startframe : int
        where to start.
    totalframe : bool or int
        'all' for all frames, int for special frames.
    show : TYPE, optional
        whether show the img. The default is False.
    out : TYPE, optional
        whether save the img. The default is False.

    Returns
    -------
    arr : numpy arr
        arr of all the img.

    """
    # 从第startframe（含）开始读（0-based），共读totalframe帧
    frame_size = W * H * 3 // 2
    plt.ion()
    yuvfilename=os.path.join(pathname,filename)
    basename = os.path.basename(yuvfilename)
    name = basename.split('.')[0]
    print("start")
    print(yuvfilename)
    with open(yuvfilename, 'rb') as fp:
        fp.seek(0,2)
        ps = fp.tell() #当前文件指针位置
        numframes = ps // frame_size
        fp.seek(startframe * frame_size, 0) #跳过前startframe帧
        if totalframe == 'all':
            totalframe = numframes
        arr = np.zeros((totalframe,H,W,3), np.uint8)
        print('reading ' + yuvfilename)
        for i in range(totalframe):
            oneframe_I420 = np.zeros((H*3//2,W),np.uint8)
            for j in range(H*3//2):
                for k in range(W):
                    oneframe_I420[j,k] = int.from_bytes(fp.read(1), byteorder='little', signed=False)
            oneframe_RGB = cv2.cvtColor(oneframe_I420,cv2.COLOR_YUV2RGB_I420)
            if show:
                plt.imshow(oneframe_RGB)
                plt.show()
                plt.pause(0.001)
            if out:
                pathname_image=os.path.join(pathname,name)
                pathname_image=pathname_image.replace("/","_image/")
                #print(pathname_image)
                if os.path.exists(pathname_image):
                    tmp='%03d' %i
                    tmp=pathname_image+'/'+name+'-'+tmp+'.png' 
                    cv2.imwrite(tmp,oneframe_RGB[:,:,::-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    os.makedirs(pathname_image) 
                    tmp='%03d' %i
                    tmp=pathname_image+'/'+name+'-'+tmp+'.png' 
                    cv2.imwrite(tmp,oneframe_RGB[:,:,::-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
            arr[i] = oneframe_RGB
    return arr
def video_image(pathname,filename):
    print("start")
    videoname=os.path.join(pathname,filename)
    print(videoname)
    cap = cv2.VideoCapture(videoname)
    basename = os.path.basename(videoname)
    name = basename.split('.')[0]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps =cap.get(cv2.CAP_PROP_FPS)
    #fps=30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #size=(960,544)
    i=0
    while(cap.isOpened()):
        i=i+1
        ret, frame = cap.read()
        if ret==True:
            pathname_image=os.path.join(pathname,name)
            pathname_image=pathname_image.replace("/","_image/")
            if os.path.exists(pathname_image):
                cv2.imwrite(pathname_image+'/'+name+'-'+str(i)+'.jpg',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                os.makedirs(pathname_image) 
                cv2.imwrite(pathname_image+'/'+name+'-'+str(i)+'.jpg',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, help='Path to mp4 file.')
    args = parser.parse_args()
    dirname=args.file
    for (root, dirs, files) in os.walk(dirname):
        print(root)
        print(dirs)
        if root!=dirname:
            print("done")
            break
        else:
            for dirp in dirs:
                pathname=os.path.join(dirname,dirp)
                for (root_f, dirs_f, files_f) in os.walk(pathname):
                    for filename in files_f:
                        if filename.endswith(".yuv"):
                            yuv2rgb(pathname,filename, 1280, 720, 0, "all", show=False, out=True)
if __name__ == '__main__':
    main()



