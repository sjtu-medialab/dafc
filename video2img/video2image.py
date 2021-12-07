import cv2
import os
import argparse
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
                cv2.imwrite(pathname_image+'/'+name+'-'+str(i)+'.png',frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                os.makedirs(pathname_image) 
                cv2.imwrite(pathname_image+'/'+name+'-'+str(i)+'.png',frame)
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
                        if filename.endswith(".mp4"):
                            video_image(pathname,filename)
if __name__ == '__main__':
    main()



