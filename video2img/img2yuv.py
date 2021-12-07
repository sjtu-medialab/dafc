
import argparse
import os
from PIL import Image
from ffmpy3 import FFmpeg

def image_yuv(yuvname,pathname,filename):
    pathname=pathname.replace('Video','Video_yuv8_')
    if os.path.exists(pathname):
        
        picpath=os.path.join(yuvname,filename+'-%d-LR.png')
        outname = os.path.join(pathname,filename+'.yuv')
        print(picpath)
        ff = FFmpeg(inputs={picpath:None},
                            outputs={outname:'-r 25 -s 320x176 -pix_fmt yuv420p'})
        print(ff.cmd)
        ff.run()
    else:
        os.makedirs(pathname)
        
        picpath=os.path.join(yuvname,filename+'-%d-LR.png')
        outname = os.path.join(pathname,filename+'.yuv')
        print(picpath)
        ff = FFmpeg(inputs={picpath:None},
                            outputs={outname:'-r 25 -s 320x176 -pix_fmt yuv420p'})
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
