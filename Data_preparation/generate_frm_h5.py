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

video_data_dir = '/wrk/yangruij/DONOTREMOVE/git/BioVid_multimodal/data/Biodevkit/cropped_frm'
out_h5py = '/wrk/yangruij/DONOTREMOVE/git/BioVid_multimodal/data/biovid_cropfrm_h5/'

def gen_frmh5(usr, vid, users, video):
    vidshape = np.shape(vid)
    usr.create_dataset(name=video, data=vid, shape=vidshape, dtype=np.uint8, compression="gzip", compression_opts=9)
    
def shape2points(shape, dtype = 'int', pointNum = 68):
    coords = np.zeros((pointNum, 2), dtype=dtype)
    for i in range(0, pointNum):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords
    
    
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
        
        frm_dir = os.listdir(inputvid_path)
        frm_dir = sorted(frm_dir)
        vid4h5 = np.zeros((138, 190, 145, 3))
        
        for f in range(len(frm_dir)):
            frm = frm_dir[f]
            frm_path = os.path.join(inputvid_path, frm)
            Img = cv2.imread(frm_path)  
            Img = cv2.resize(Img, (145, 190))
            print(Img.shape)
            vid4h5[f,:,:,:] = Img                                    
            
        gen_frmh5(dataset, vid4h5, users, video) 

    print('Processing One person takes {}'.format(time.time()-t1))
  
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--part', type=int, default=1, help='the gpu id used for predict')
    parser.add_argument('--num_workers', type=int, default=40, help='initial learning rate') 
    args = parser.parse_args()
    
    num_worker = args.num_workers
    pool = Pool(num_worker)
    users = os.listdir(video_data_dir)
    users = sorted(users)

    #if not os.path.exists(orig_vispath):
        #os.makedirs(orig_vispath, exist_ok=True)
    
    if not os.path.exists(out_h5py):
        os.makedirs(out_h5py, exist_ok=True)
        
    if args.part == 1:
        users = users[:40]
    elif args.part == 2:
        users = users[40:]

    pool.map(crop_face, users)
   
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
