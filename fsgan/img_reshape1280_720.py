
import argparse
import os
from PIL import Image
import numpy as np


def image_yuv(yuvname,pathname,filename):
    pathname=yuvname.replace('val','vald')
    for (root, dirs, files) in os.walk(yuvname):
        for filename in files:
            if os.path.exists(pathname):
                img = Image.new('RGB', (1280, 720), (255, 255, 255))
                p_filename=os.path.join(yuvname,filename) 
                pil_im = Image.open(p_filename) 
                img= np.array(img) 
                pil_im=np.array(pil_im)
                img[:,280:1000,:]=pil_im  
                img_tr = Image.fromarray(img) 
                #out = img_tr.resize((320,180))   
                img_tr.save(os.path.join(pathname,filename))
            else:
                os.makedirs(pathname)
                img = Image.new('RGB', (1280, 720), (255, 255, 255))
                p_filename=os.path.join(yuvname,filename) 
                pil_im = Image.open(p_filename) 
                img= np.array(img) 
                pil_im=np.array(pil_im)
                img[:,280:1000,:]=pil_im  
                img_tr = Image.fromarray(img) 
                #out = img_tr.resize((320,180))   
                img_tr.save(os.path.join(pathname,filename))




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
                    if root_f!=pathname:
                        print("subfile_done")
                        break
                    else:
                        for filename in dirs_f:
                            yuvname=os.path.join(pathname,filename)
                            
                            image_yuv(yuvname,pathname,filename)
if __name__ == '__main__':
    main()



#ffmpeg -f rawvideo -vcodec rawvideo -s 260x358 -r 25 -pix_fmt yuv420p -i /home/dafc/data/dev/projects/fsgan/Video_yuv8__x4__val_image/Actor_24_image/02-02-06-02-02-02-24.yuv -c:v libx264 -preset ultrafast -qp 0 output.mp4
#python test_FaceDict.py --test_path /home/dafc/DFDNet-whole/Val_image_x4p_decode_256_176/Actor_05_image_x4p/01-01-02-02-01-01-05 --results_dir ./Results/TestWholeResults --upscale_factor 1 --gpu_ids 0