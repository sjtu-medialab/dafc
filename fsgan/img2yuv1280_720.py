
import argparse
import os
from PIL import Image
from ffmpy3 import FFmpeg

def image_yuv(yuvname,pathname,filename):
    pathname=pathname.replace('Video','Video_yuv8_')
    if os.path.exists(pathname):
        
        picpath=os.path.join(yuvname,filename+'-%3d.png')
        outname = os.path.join(pathname,filename+'.yuv')
        outname1 = os.path.join(pathname,filename+'.mp4')
        print(picpath)
        ff = FFmpeg(inputs={picpath:None},
                            outputs={outname:'-r 25 -s 1280x720 -pix_fmt yuv420p'})
        print(ff.cmd)
        ff.run()
        ff = FFmpeg(inputs={outname:'-f rawvideo -vcodec rawvideo -s 1280x720 -r 25 -pix_fmt yuv420p'},
                            outputs={outname1:'-c:v libx264 -preset ultrafast -qp 0'})
        print(ff.cmd)
        ff.run()
    else:
        os.makedirs(pathname)
        
        picpath=os.path.join(yuvname,filename+'-%3d.png')
        outname = os.path.join(pathname,filename+'.yuv')
        outname1 = os.path.join(pathname,filename+'.mp4')
        print(picpath)
        ff = FFmpeg(inputs={picpath:None},
                            outputs={outname:'-r 25 -s 1280x720 -pix_fmt yuv420p'})
        print(ff.cmd)
        ff.run()
        ff = FFmpeg(inputs={outname:'-f rawvideo -vcodec rawvideo -s 1280x720 -r 25 -pix_fmt yuv420p'},
                            outputs={outname1:'-c:v libx264 -preset ultrafast -qp 0'})
        print(ff.cmd)
        ff.run()
    '''
    for (root, dirs, files) in os.walk(yuvname):
        for filei in files:
            if os.path.exists(pathname):
                
                picpath = os.path.join(yuvname,filei)
                img = Image.open(picpath)
                in_wid,in_hei = img.size
                out_wid = in_wid//2*2
                out_hei = in_hei//2*2
                size = '{}x{}'.format(out_wid,out_hei)
                outname = os.path.join(pathname,filename)
                outname=outname+'.yuv'
                ff = FFmpeg(inputs={picpath:None},
                        outputs={outname:'-s {} -pix_fmt yuv420p'.format(size)})
                
                ff.run()
                
            else:
                
                os.makedirs(pathname) 
                picpath = os.path.join(yuvname,filesi)
                img = Image.open(picpath)
                in_wid,in_hei = img.size
                out_wid = in_wid//2*2
                out_hei = in_hei//2*2
                size = '{}x{}'.format(out_wid,out_hei)
                outname = os.path.join(pathname,filename)
                outname=outname+'.yuv'
                ff = FFmpeg(inputs={picpath:None},
                        outputs={outname:'-s {} -pix_fmt yuv420p'.format(size)})
                
                ff.run()
        
'''

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
