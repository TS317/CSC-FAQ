'''
Created on Oct 7, 2019

@author: ryang
'''
import torch
import torchvision
import torch.utils.data as data
from PIL import Image
import os
import os.path as osp
import math
import functools
import copy 
import numpy as np
from setuptools import namespaces
from PIL.ImageTransform import Transform
from torchvision.transforms import transforms
import pandas as pd
import pdb
import math
from numpy.core.defchararray import endswith
import cv2
import random
import h5py
from sklearn.manifold.mds import smacof
import matplotlib.pyplot as plt
import pylab
from vision import vis_face
from torch.utils.data import DataLoader

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']
        new_video_x = (video_x - 127.5)/128
        return {'video_x': new_video_x, 'pain_label': pain_label}

class unNormalization(object):
    
    def __call__(self, sample):
        new_video_x = sample*128+127.5 # new = (sample-(-127.5/128))/(1/128)
        
        return new_video_x
        
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size=(125, 100)):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']

        clip_frames, h, w = video_x.shape[0], video_x.shape[1], video_x.shape[2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        new_video_x = np.zeros((clip_frames, new_h, new_w, 3))
        for i in range(clip_frames):
            image = video_x[i, :, :, :]
            #img = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
            img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            new_video_x[i, :, :, :] = img

        return {'video_x': new_video_x, 'pain_label': pain_label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size=(125, 100)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']

        clip_frames, h, w = video_x.shape[0], video_x.shape[1], video_x.shape[2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        new_video_x = np.zeros((clip_frames, new_h, new_w, 3))
        for i in range(clip_frames):
            image = video_x[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video_x': new_video_x, 'pain_label': pain_label}


class CenterCrop(object):
    def __init__(self, output_size=(125, 100)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']

        clip_frames, h, w = video_x.shape[0], video_x.shape[1], video_x.shape[2]
        new_h, new_w = self.output_size

        top = int(round((h - new_h) /2.))
        left = int(round((w - new_w) / 2.))

        new_video_x = np.zeros((clip_frames, new_h, new_w, 3))
        
        for i in range(clip_frames):
            image = video_x[i, :, :, :]
            image = image[top: top + new_h, left: left + new_w]
            new_video_x[i, :, :, :] = image

        return {'video_x': new_video_x, 'pain_label': pain_label}
        

class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']
        clip_frames, h, w = video_x.shape[0], video_x.shape[1], video_x.shape[2]
        new_video_x = np.zeros((clip_frames, h, w, 3))

        p = random.random()
        if p < 0.5:
            #print('Flip')
            for i in range(clip_frames):
                # video 
                image = video_x[i, :, :, :]
                image = cv2.flip(image, 1)
                new_video_x[i, :, :, :] = image  
                
            return {'video_x': new_video_x, 'pain_label': pain_label}
        else:
            #print('no Flip')
            return {'video_x': video_x, 'pain_label': pain_label}



class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        video_x, pain_label = sample['video_x'], sample['pain_label']
        video_x = video_x.transpose((3, 0, 1, 2))
        video_x = np.array(video_x)
      
        pain_label_np = np.array([0],dtype=np.long)
        pain_label_np[0] = pain_label

        return {'video_x': torch.from_numpy(video_x.astype(np.float)).float(), 'pain_label': torch.from_numpy(pain_label_np.astype(np.long)).long()}


class BioVid(data.Dataset):#base_path-5fold name 
    def __init__(self, info_list, root_path, sample_duration, transform=None):
        self.root_path = root_path
        self.info_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.video_list = [self.info_frame.iloc[i, 0] for i in range(len(self.info_frame))]
        self.label_list = [self.info_frame.iloc[i, 1] for i in range(len(self.info_frame))]
        self.sample_duration = sample_duration
        
        self.transform = transform

    def __len__(self):
        return len(self.info_frame) #num of videos

    def __getitem__(self, idx):
        #load video, then segment here
        pain_label = self.label_list[idx]
        vidpathname = self.video_list[idx]
        usrname, vidname = vidpathname.split('/',1)
        #print(vidname)
        self.database = h5py.File(os.path.join(self.root_path, usrname+'.hdf5'), 'r')
        video = self.database[vidname]
        
        len_of_frames = np.shape(video)[0] #len(video)
        #randome select one segment from a video
        if  len_of_frames > self.sample_duration:
            i_s = np.random.randint(len_of_frames-self.sample_duration-1)
            i_e = i_s + self.sample_duration
        else:
            print('video length is too short!')
            i_s = 0
            i_e = len_of_frames - 1

        clip = video[i_s:i_e, :, :, :]
        sample = {'video_x': clip, 'pain_label': pain_label}

        if self.transform:
            sample = self.transform(sample)
        
        sample_all = copy.deepcopy(sample)
        return sample_all

if __name__ == '__main__':
    
    root_list = '../BioVid/biovid_original_frm_h5'
    trainval_list = '../test_T0T4_fold1' + '.txt' #+'%d' % (index)
    orig_vispath = '../071309'
    
    for fold in range(1):
        fold_num = str(fold + 1)
    
        data = BioVid(trainval_list, root_list, 32, transform=transforms.Compose([Normaliztion(), Rescale((230,300)), RandomCrop((224,224)),RandomHorizontalFlip(),  ToTensor()]))
        train_loader = DataLoader(data, batch_size=3, shuffle=True, num_workers=3, drop_last=True)
        untransform = unNormalization()

        for i, sample_batched in enumerate(train_loader):
            print(i, sample_batched['video_x'].shape, sample_batched['pain_label'])

            imgarray = sample_batched['video_x'][1,:,5,:,:].squeeze()
            imgarray = untransform(imgarray)
            imgarray = imgarray.cpu().numpy().astype('uint8')
            imgarray = np.transpose(imgarray, (1,2,0))
            img_bg = cv2.cvtColor(imgarray, cv2.COLOR_BGR2RGB)
            
            figure = pylab.figure()
            # plt.subplot(121)
            pylab.imshow(img_bg)
            pylab.savefig(orig_vispath)
