import sys
import os
import warnings
import glob
from multiprocessing import Process

import numpy as np
import h5py as h5
from tqdm import tqdm
from PIL import Image
import cv2
import argparse

def downsample_gaussian_blur(img,ratio):
    sigma=(1/ratio)/3
    # ksize=np.ceil(2*sigma)
    ksize=int(np.ceil(((sigma-0.8)/0.3+1)*2+1))
    ksize=ksize+1 if ksize%2==0 else ksize
    img=cv2.GaussianBlur(img,(ksize,ksize),sigma,borderType=cv2.BORDER_REFLECT101)
    return img

def resize_img(img_in, ratio, crop = False):
    # if ratio>=1.0: return img
    tag = False
    if type(img_in)==np.array:
        tag = True
        img = cv2.cvtColor(np.array(img_in),cv2.COLOR_RGB2BGR)
    else:
        img = img_in
    # crop the meaningless border
    if crop:
        img = img[int(img.shape[0] * 0.1): int(img.shape[0] * 0.9),0:int(img.shape[1])]
    h, w, _ = img.shape
    hn, wn = int(np.round(h * ratio)), int(np.round(w * ratio))
    img_out = cv2.resize(downsample_gaussian_blur(img, ratio), (wn, hn), cv2.INTER_LINEAR)
    if tag:
        img_out = Image.fromarray(cv2.cvtColor(img_out,cv2.COLOR_BGR2RGB))
    return img_out

# execute a task includes all operation for a single image
def task(image_path:str, target_path:str, ratio:float):
    image_name = os.path.basename(image_path)
    img = cv2.imread(image_path)
    img = resize_img(img,ratio,crop=True)
    img = resize_img(img,ratio)
    image_out_path = os.path.join(target_path,image_name)
    cv2.imwrite(image_out_path,img)

def main():
    parser = argparse.ArgumentParser(description='panorama image preprocess')
    parser.add_argument('--dataset', default='KITTI360', type=str,
                        help='KITTI360, WUHAN, SHANGHAI')
    parser.add_argument('--raw_path', type=str, required=True, default='/data-lyh2/KITTI360',
                        help='Base path storing whole data base, for KITTI360, or others')
    parser.add_argument('--target_pano_path', type=str, required=True, default='/data-lyh2/KITTI360',
                        help='The out put preprocssed panorama images, default the same base path')
    parser.add_argument('--sequence', default='3', type=int,
                        help='0, 1 and etc, for single_sequence test')
    parser.add_argument('--single_sequence', action='store_true', 
                        help='process single sequence images for debug')
    parser.add_argument('--single_image', action='store_true', 
                        help='process single image for visualization')
    parser.add_argument('--multi_thread', action='store_true', 
                        help='process single thread')
    args = parser.parse_args()
    print(args)
    
    # define the source image path and target image path
    kitti360panoPath = args.raw_path
    kitti360panoh5Path = args.target_pano_path
    
    if args.single_image:
        if args.dataset=='KITTI360':
            seq_all = [3]
            sequence = '2013_05_28_drive_%04d_sync'%seq_all[0]
            target_path = os.path.join(kitti360panoPath,"data_2d_pano",sequence)
            seq_key = os.path.join(target_path, 'pano', 'data_rgb', '%010d' % 0 + '.png') 
            img_pil = Image.open(seq_key).convert('RGB')
            img_pil.save('./%010d' % 0 + '_raw.png')
            img = resize_img(img_pil,0.5, crop=True)
            img = resize_img(img,0.5)
            img.save('./%010d' % 0 + '_resize.png')
        elif args.dataset=='WUHAN':
            set = 1
        elif args.dataset=='SHANGHAI':
            set = 2
        return
    
    if args.single_sequence:
        if args.dataset=='KITTI360':
            seq_all = [args.sequence]
        elif args.dataset=='WUHAN':
            set = 1
        elif args.dataset=='SHANGHAI':
            set = 2
    else:
        if args.dataset=='KITTI360':
            #seq_all = [0,2,3,4,5,6,7,9,10]
            seq_all = [2,4,5,6,7,9,10]
        elif args.dataset=='WUHAN':
            set = 1
        elif args.dataset=='SHANGHAI':
            set = 2
    
    # loop over whole sequences
    for seq in tqdm(seq_all,desc='all sequences'.rjust(15)):
        sequence = '2013_05_28_drive_%04d_sync'%seq      
        # loop over frames
        source_path = os.path.join(kitti360panoPath,"data_2d_pano",sequence)
        if not os.path.exists(source_path):
            warnings.warn('You have chosen a nonesisted image path')
        target_path = os.path.join(kitti360panoh5Path,"data_2d_pano_h5",sequence,'pano','data_rgb')
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        # glob images
        images = glob.glob(os.path.join(source_path, 'pano', 'data_rgb') + '/*.png')
        # single thread sequential
        if not args.multi_thread:
            for image in tqdm(images,desc='single sequence'.rjust(15)):
                image_name = os.path.basename(image)
                img = cv2.imread(image)
                img = resize_img(img,0.5,crop=True)
                img = resize_img(img,0.5)
                image_out_path = os.path.join(target_path,image_name)
                cv2.imwrite(image_out_path,img)
        else:
            warnings.warn('pretty slow than single thread, fixme')
            # multi thread parallel
            # # create all tasks
            # processes = [Process(target=task, args=(image,target_path,0.5)) for image in images]
            # # start all processes
            # for process in processes:
            #     process.start()
            # # wait for all processes to complete
            # for process in processes:
            #     process.join()


if __name__=="__main__":
    print('Begin!')
    main()
    print('Done!', flush=True)