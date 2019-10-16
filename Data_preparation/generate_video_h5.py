'''
Created on Jun 5, 2019

@author: ryang
'''

import dlib
import cv2
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import random
import pylab
from matplotlib.patches import Circle
from multiprocessing import Pool
import scipy.io as sio
import h5py
import time

video_data_dir = '../data/video'
out_h5py = '../data/biovid_origfrm_h5/'


def gen_frmh5(usr, vid, users, video):
    vidshape = np.shape(vid)
    usr.create_dataset(name=video[:-4], data=vid, shape=vidshape, dtype=np.int8, compression="gzip", compression_opts=9)

    
def crop_face(aug):
    facenotdet = []
    users = aug
    
    t1 = time.time()
    vid_path = os.path.join(video_data_dir, users)
    vid_dir = os.listdir(vid_path)
    vid_dir = sorted(vid_dir)
    
    
    dataset = h5py.File(out_h5py + users + '.hdf5', 'a')
    #vid_ds = dataset.create_group('video_data')
    
    for v in range(len(vid_dir)):           
        video = vid_dir[v]
        inputvid_path = os.path.join(vid_path, video)
   
        vidObj = cv2.VideoCapture(inputvid_path)
        count = 0
        ldmk = []

        vid4h5 = np.zeros((138, 190, 145, 3))      
        
        while (vidObj.isOpened()):
            success, img = vidObj.read()
            if success == False:
                break
            count += 1 
            
            Img = cv2.resize(img, (145, 190))
            vid4h5[count-1,:,:,:] = Img                                    
            
        gen_frmh5(dataset, vid4h5, users, video) 

    print('Processing One person takes {}'.format(time.time()-t1))
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--part', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--num_workers', type=int, default=80, help='initial learning rate') 
    args = parser.parse_args()
    
    num_worker = args.num_workers
    pool = Pool(num_worker)
    users = os.listdir(video_data_dir)
    users = sorted(users)

    if not os.path.exists(orig_vispath):
        os.makedirs(orig_vispath, exist_ok=True)
    
    if not os.path.exists(out_h5py):
        os.makedirs(out_h5py, exist_ok=True)
        
    if args.part == 1:
        users = users[:40]
    elif args.part == 2:
        users = users[40:]

    pool.map(crop_face, users)
    #face_cascade = cv2.CascadeClassifier('/home/ryang/anaconda3/envs/ptenv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    
#     # read hdf5
#     hdf5File = h5py.File(out_h5py+'.hdf5', "r")
#     for gname, group in hdf5File.items():
#         print(gname)
#         for dname, ds in group.items():           
#             print(dname)  
#             for dataname, data in ds.items():
#                 print(dataname)
#                  
#     vidshape = hdf5File[os.path.join('video_data', dname, dataname)].shape
#     imgarray = hdf5File[os.path.join('video_data', dname, dataname)][5,:,:,:]
#     imgarray = imgarray.astype('uint8')
#     img_bg = cv2.cvtColor(imgarray, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_bg)
#     plt.savefig(orig_vispath)
