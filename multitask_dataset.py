"""

"""

import os
import numpy as np
import random 
import _pickle as pickle
import torch
import torchvision.transforms.functional as F
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms
import time
import re

def pil_to_cv(pil_im):
    """ convert PIL image to cv2 image"""
    open_cv_image = np.array(pil_im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1] 


def plot_text(im,offset,cls,idnum,class_colors,class_dict):
    """ Plots filled text box on original image, 
        utility function for plot_bboxes_2
        im - cv2 image
        offset - to upper left corner of bbox above which text is to be plotted
        cls - string
        class_colors - list of 3 tuples of ints in range (0,255)
        class_dict - dictionary that converts class strings to ints and vice versa
    """

    text = "{}: {}".format(idnum,class_dict[cls])
    
    font_scale = 2.0
    font = cv2.FONT_HERSHEY_PLAIN
    
    # set the rectangle background to white
    rectangle_bgr = class_colors[cls]
    
    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    
    # set the text start position
    text_offset_x = int(offset[0])
    text_offset_y = int(offset[1])
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(im, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(im, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0., 0., 0.), thickness=2)



class I24_Dataset(data.Dataset):
    """
    Returns 3D labels and images for 3D detector training
    """
    
    def __init__(self, 
                 dataset_dir, 
                 label_format = "tailed_footprint", 
                 mode = "train",
                 CROP = 0,
                 multiple_frames = False,
                 random_partition = False,
                 mask_dir = None):
        """ 
        dataset_dir - (str) path to dataset created by export_dataset.py
        label_format - (str) tailed_footprint or 8-corners I think
        mode  - (str) "train" or "validation"
        CROP - (int) if 0, no cropping, else positive integer size of crops (either 224 or 112 generally)
        
        multiple_frames - (bool) load and use multiple frames in dataset
        random_partition - (bool) is train val split random or the tail end of each scene
        mask_dir (str) if not None, path to masking images to be used to mask irrelevant portions of each image (reduce FP)
        """
        
        # load masks
        self.mask_ims = None
        if mask_dir is not None:

            self.mask_ims = {1: {},
                         2: {},
                         3: {},
                         999: {}}
        
            for scene_id in [1,2,3,999]:
                scene_mask_dir = os.path.join(mask_dir,"scene{}".format(scene_id))
                mask_paths = os.listdir(scene_mask_dir)
                for path in mask_paths:
                    if "1080"  in path:
                        key = path.split("_")[0]
                        path = os.path.join(scene_mask_dir, path)
                        im = cv2.imread(path)
        
                        self.mask_ims[scene_id][key] = im
        
        #torch.random.manual_seed(0)
        random.seed(0)
        
        self.mode = mode
        self.label_format = label_format
        self.CROP = CROP
        self.multiple_frames = multiple_frames
        
        self.im_tf = transforms.Compose([
                # transforms.RandomApply([
                #     transforms.ColorJitter(brightness = 0.6,contrast = 0.6,saturation = 0.5)
                #         ]),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.05), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),
                # transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=(0.485,0.456,0.406)),

                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
                ])

        # for denormalizing
        self.denorm = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                           std = [1/0.229, 1/0.224, 1/0.225])
        
         
        
        
        # self.classes = { "sedan":0,
        #             "midsize":1,
        #             "van":2,
        #             "pickup":3,
        #             "semi":4,
        #             "truck (other)":5,
        #             "truck": 5,
        #             "motorcycle":6,
        #             "trailer":7,
        #             0:"sedan",
        #             1:"midsize",
        #             2:"van",
        #             3:"pickup",
        #             4:"semi",
        #             5:"truck (other)",
        #             6:"motorcycle",
        #             7:"trailer",
        #             }
        
        self.classes = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi_sleeper":4,
                    "semi_nonsleeper":5,
                    "truck": 6,
                    "motorcycle":7,

                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi_sleeper",
                    5:"semi_nonsleeper",
                    6:"truck",
                    7:"motorcycle",
                    }
        
        self.class_colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,0),
            (255,0,255),
            (0,255,255),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50)]
            
        self.labels = []
        self.data = []
        self.ids = []
        self.state_labels = []

        # an intermediate file that combines all labels into a single file
        complete_cache_file = os.path.join(dataset_dir,"dataset_cache.cpkl")
        
        try:        
            with open(complete_cache_file,"rb") as f:
                [self.data,self.labels,self.ids,self.state_labels] = pickle.load(f)
            
        except:
            
            # load label file and parse
            dsp = os.path.join(dataset_dir,"data_summary.cpkl")
            with open(dsp,"rb") as f:
                self.data_summary = pickle.load(f)
            
            
            random.shuffle(self.data_summary)
            
            for idx, item in enumerate(self.data_summary):
                if idx % 1000 == 0: 
                    print("Loaded label {} of {}".format(idx,len(self.data_summary)))
                
                single_label_file = item[1]
                
                with open(single_label_file,"rb") as f:
                    frame_labels = pickle.load(f) 
                
                EXCLUDE = False
                frame_boxes = []
                frame_ids = []
                frame_states = []
                if len(frame_labels) == 0:
                    frame_boxes = [torch.zeros(22)]
                    frame_ids = [torch.zeros(1)-1]
                    frame_states = [torch.zeros(6)]
    
                else:
                    for box in frame_labels:
                                            
                        # if box["gen"] == "Spline Interp":
                        #     continue
                        
                        # store object class
                        try:
                            if "semi" in box["class"]:
                                box["class"] = box["class"].split("_")[0] + "_" + box["class"].split("_")[1]
                            elif "truck" in box["class"]:
                                box["class"] = box["class"].split("_")[0] 
                            cls = torch.ones([1])* self.classes[box["class"]]
                        except:
                            cls = torch.zeros([1])
                        
                        # store 3D (im space) bbox
                        try:
                            bbox3d = box["im_box"].float().reshape(-1) # need to reshape to correct shape
                        except:
                            EXCLUDE = True
                        
                        # store 2D bbox
                        bbox2d = torch.zeros([4])
                        bbox2d[0] = torch.min(bbox3d[::2])
                        bbox2d[1] = torch.min(bbox3d[1::2])
                        bbox2d[2] = torch.max(bbox3d[::2])
                        bbox2d[3] = torch.max(bbox3d[1::2])
                        
                        state = box["box"]
                        id = torch.tensor(99).unsqueeze(0)# Doesn't matter any more since we're not doing reid      torch.ones([1])*int(str(box["id"]) + str(single_label_file.split("scene_")[1][0])) # append scene id to make ids unique across all scenes
                        #reformat label so each frame is a tensor of size [n objs, label_format_length + 1] where +1 is class index
                        bbox = torch.cat((bbox3d,bbox2d,id,cls),dim = 0).float()
                        #bbox = torch.from_numpy(bbox)
                        frame_boxes.append(bbox)
                        frame_ids.append(id)
                        frame_states.append(state)
                
                if not EXCLUDE:
                    try:
                        frame_boxes = torch.stack(frame_boxes)
                        frame_ids = torch.stack(frame_ids)
                        frame_states = torch.stack(frame_states)
                    except:
                        pass
                    
                    self.data.append(item[0])
                    self.labels.append(frame_boxes)
                    self.ids.append(frame_ids)
                    self.state_labels.append(frame_states)
                # if idx > 500:
                #     break
    
            
            if True:
                with open(complete_cache_file,"wb") as f:
                    pickle.dump([self.data,self.labels,self.ids,self.state_labels],f)
            

        # partition dataset
        if random_partition:
            if self.mode == "train":
                self.data = self.data[:int(len(self.data)*0.9)]
                self.labels = self.labels[:int(len(self.labels)*0.9)]
                self.ids = self.ids[:int(len(self.ids)*0.9)]
                self.state_labels = self.state_labels[:int(len(self.state_labels)*0.9)]

            else:
                self.data = self.data[int(len(self.data)*0.9):]
                self.labels = self.labels[int(len(self.labels)*0.9):]
                self.ids = self.ids[int(len(self.ids)*0.9):]
                self.state_labels = self.state_labels[int(len(self.state_labels)*0.9):]
                
        else: # hold back the last 1-ratio percent of each scene
            ratio = 0.8
            scene_validation_frame = {
                1:int(2700*ratio),
                2:int(1800*ratio),
                3:int(1800*ratio)
                }
            
            # get indices of all data that falls within these portions of each scene
            train_idxs = []
            for d_idx,path in enumerate(self.data):
                scene = int(path.split("/")[-3].split("_")[-1])
                idx   = int(path.split("/")[-1].split(".")[0])
                if idx < scene_validation_frame[scene]:
                    train_idxs.append(d_idx)
            
            if self.mode == "train":
                data   = [self.data[idx]   for idx in train_idxs]
                labels = [self.labels[idx] for idx in train_idxs]
                ids    = [self.ids[idx]    for idx in train_idxs]
                states = [self.state_labels[idx] for idx in train_idxs]
                
            else:
                val_idxs = []
                for i in range(len(self.data)):
                    if i not in train_idxs:
                        val_idxs.append(i)
                data   = [self.data[idx]   for idx in val_idxs]
                labels = [self.labels[idx] for idx in val_idxs]
                ids    = [self.ids[idx]    for idx in val_idxs]
                states = [self.state_labels[idx] for idx in val_idxs]
                
            self.data = data
            self.labels = labels
            self.ids = ids
            self.state_labels = states
        
    
    def __getitem__(self,index,crop_index = None):
        """ returns item indexed from all frames in all tracks from training
        or testing indices depending on mode
        """
        no_labels = False
        
        # load image and get label        
        y = self.labels[index].clone()
        im = Image.open(self.data[index])
        

        
        if self.multiple_frames:
            try:
                im_path = self.data[index]
                im_idx = int(im_path.split(".")[0].split("/")[-1])
                shift = np.random.randint(1,5)
                im_idx -= shift
                if im_idx < 0:
                    im_idx = 0
                im_idx = str(im_idx).zfill(4)
                prev_path = im_path
                
                im_idx += ".png"
                replace = len(im_idx)
                prev_path = "/".join((prev_path.split("/")[:-1] + [im_idx]))
            except:
                prev_path = im_path
            if not os.path.exists(prev_path):
                prev_path = im_path
            prev_im = Image.open(prev_path)
        # camera_id = self.data[index].split("/")[-1].split("_")[0]
        # vps = self.vps[camera_id]
        # vps = torch.tensor([vps[0][0],vps[0][1],vps[1][0],vps[1][1],vps[2][0],vps[2][1]])
        
        #mask_regions = self.box_2d[index]
        
        
        
        if self.mask_ims is not None:
            
            # convert each to array to mask it
            np_im = np.array(im).astype(float)
            
            try:
                camera = self.data[index].split("/")[-2]
                scene_id = int(self.data[index].split("/")[-3].split("_")[-1])
            except:
                camera = self.data[index].split("/")[-1].split("_")[0]
                scene_id = 999
            
            # get mask im
            mask_im = self.mask_ims[scene_id][camera]/255
            blur_im = cv2.blur(np_im, (17, 17)).astype(float)
            
            np_im = np_im*mask_im + blur_im * (1-mask_im)

            im = Image.fromarray(np_im.astype(np.uint8))
            
            if self.multiple_frames:
                prev_np_im = np.array(prev_im).astype(float)
                blur_prev_im = cv2.blur(prev_np_im, (17, 17)).astype(float)
                prev_np_im = prev_np_im*mask_im + blur_prev_im * (1-mask_im)
                prev_im = Image.fromarray(prev_np_im.astype(np.uint8))

        
        if y.numel() == 0:
            y = torch.zeros([1,22]) -1
            no_labels = True
            
        # I don't think this is relevant any more
        # elif camera_id in ["p2c2","p2c3","p2c4"]:
        #     new_y = torch.clone(y)
        #     new_y[:,[0,2,4,6,8,10,12,14,16,18]] = y[:,[2,0,6,4,10,8,14,12,18,16]] # labels are expected left first then right, but are formatted right first
        #     new_y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[3,1,7,5,11,9,15,13,19,17]]
        #     y = new_y
            
            # inspect each - if right side is closer to vanishi
        
        # im = F.to_tensor(im)
        
        # for region in mask_regions:
        #     im[:,region[1]:region[3],region[0]:region[2]] = 0
            
        # im = F.to_pil_image(im)
        
        if self.mode == "train":
            # stretch and scale randomly by a small amount (0.8 - 1.2 x in either dimension)
            scale = max(1,np.random.normal(1,0.1))
            aspect_ratio = max(0.75,np.random.normal(1,0.2))
            size = im.size
            new_size = (int(im.size[1] * scale * aspect_ratio),int(im.size[0] * scale))
            im = F.resize(im,new_size)
            im = F.to_tensor(im)
            
            if self.multiple_frames:
                prev_im = F.resize(prev_im,new_size)
                prev_im = F.to_tensor(prev_im)
                im = torch.cat((im,prev_im),dim = 0)
            
            new_im = torch.rand([im.shape[0],size[1],size[0]])
            new_im[:,:min(im.shape[1],new_im.shape[1]),:min(im.shape[2],new_im.shape[2])] = im[:,:min(im.shape[1],new_im.shape[1]),:min(im.shape[2],new_im.shape[2])]
                    
            if self.multiple_frames:
                im = F.to_pil_image(new_im[:3,:,:])
                prev_im = F.to_pil_image(new_im[3:,:,:])
            else: im = F.to_pil_image(new_im)
            
            y[:,[0,2,4,6,8,10,12,14,16,18]] = y[:,[0,2,4,6,8,10,12,14,16,18]] * scale 
            y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[1,3,5,7,9,11,13,15,17,19]] * scale * aspect_ratio
            # vps[[0,2,4]] = vps[[0,2,4]] * scale
            # vps[[1,3,5]] = vps[[1,3,5]] * scale * aspect_ratio
            
            #randomly flip
            FLIP = np.random.rand()
            if FLIP > 0.5:
                im= F.hflip(im)
                if self.multiple_frames: prev_im = F.hflip(prev_im)
                # reverse coords and also switch xmin and xmax
                new_y = torch.clone(y)
                #new_y[:,[0,2,4,6,8,10,12,14,16,18]] = im.size[0] - y[:,[0,2,4,6,8,10,12,14,16,18]]
                new_y[:,[0,2,4,6,8,10,12,14,16,18]] = im.size[0] - y[:,[2,0,6,4,10,8,14,12,18,16]] # labels are expected left first then right, but are formatted right first
                new_y[:,[1,3,5,7,9,11,13,15,17,19]] = y[:,[3,1,7,5,11,9,15,13,19,17]]
                y = new_y
                
                # new_vps = torch.clone(vps)
                # vps[[0,2,4]] = im.size[0] - new_vps[[0,2,4]]
                
                if no_labels:
                    y = torch.zeros([1,22]) -1
            
            
            # randomly rotate
            angle = (np.random.rand()*40)-20
            im = F.rotate(im, angle, interpolation = Image.BILINEAR)
            if self.multiple_frames: prev_im = F.rotate(prev_im, angle, interpolation = Image.BILINEAR)
            if not no_labels:
                # decompose each point into length, angle relative to center of image
                y_mag = torch.sqrt((y[:,::2][:,:-1] - im.size[0]/2.0)**2 + (y[:,1::2][:,:-1] - im.size[1]/2.0)**2)
                y_theta = torch.atan2((y[:,1::2][:,:-1] - im.size[1]/2.0),(y[:,::2][:,:-1] - im.size[0]/2.0))
                y_theta -= angle*(np.pi/180.0)
                
                y_new = torch.clone(y)
                y_new[:,::2][:,:-1]  = y_mag * torch.cos(y_theta)
                y_new[:,1::2][:,:-1] = y_mag * torch.sin(y_theta)
                y_new[:,::2][:,:-1]  += im.size[0]/2.0
                y_new[:,1::2][:,:-1] += im.size[1]/2.0
                y = y_new
                
                
                xmin = torch.min(y[:,::2][:,:-1],dim = 1)[0].unsqueeze(1)
                xmax = torch.max(y[:,::2][:,:-1],dim = 1)[0].unsqueeze(1)
                ymin = torch.min(y[:,1::2][:,:-1],dim = 1)[0].unsqueeze(1)
                ymax = torch.max(y[:,1::2][:,:-1],dim = 1)[0].unsqueeze(1)
                bbox_2d = torch.cat([xmin,ymin,xmax,ymax],dim = 1)
                y[:,16:20] = bbox_2d
            # now, rotate each point by the same amount
            
            # remove all labels that fall fully outside of image now
            keep = []
            for item in y:
                if min(item[[0,2,4,6,8,10,12,14,16,18]]) < im.size[0] and max(item[[0,2,4,6,8,10,12,14,16,18]]) >= 0 and min(item[[1,3,5,7,9,11,13,15,17,19]]) < im.size[1] and max(item[[1,3,5,7,9,11,13,15,17,19]]) >= 0:
                    keep.append(item)
           
            try:
                y = torch.stack(keep)
            except:
                y = torch.zeros([1,22]) -1
            
        # if self.label_format == "tailed_footprint":
        #     # average top 4 points and average bottom 4 points to get height vector
        #     bot_y = (y[:,1] + y[:,3] + y[:,5] + y[:,7])/4.0
        #     bot_x = (y[:,0] + y[:,2] + y[:,4] + y[:,6])/4.0
        #     top_x = (y[:,8] + y[:,10] + y[:,12] + y[:,14])/4.0
        #     top_y = (y[:,9] + y[:,11] + y[:,13] + y[:,15])/4.0  
        #     y_tail = top_y - bot_y
        #     x_tail = top_x - bot_x
            
        #     new_y = torch.zeros([len(y),11])
        #     new_y[:,:8] = y[:,:8]
        #     new_y[:,8] = x_tail
        #     new_y[:,9] = y_tail
        #     new_y[:,10] = y[:,-1]
        #     y = new_y
        
        
        
        
        if self.CROP == 0:
            # convert image to tensor
            im_t = self.im_tf(im)
            
            mag = 1 * torch.rand(1).item()
            bias = 3* (torch.rand(1).item() - 0.5)
            randoms = (torch.rand(im_t.shape) * mag - 0.5*mag) + bias
            
            if self.mode == "train":
                im_t += randoms
            #t = F.adjust_contrast  (t,randoms[1] + 0.7)
            #t = F.adjust_brightness(t,randoms[0] + 0.7)
            #t = F.adjust_saturation(t,randoms[2] * 2 + 0.5)
            #t = F.adjust_hue       (t,randoms[3] * 0.1 - 0.05)

            if self.multiple_frames:
                prev_im_t = self.im_tf(prev_im)
                
                if self.mode == "train":
                    randoms = (torch.rand(im_t.shape) * mag - 0.5*mag) + bias
                    prev_im_t += randoms
                
                im_t = torch.cat((im_t,prev_im_t),dim = 0)
                
                
            if self.mode == "train":
                TILE = np.random.rand()
                if TILE > 0.25:
                    # find min and max x coordinate for each bbox
                    occupied_x = []
                    occupied_y = []
                    for box in y:
                        xmin = min(box[[0,2,4,6,8,10,12,14]])
                        xmax = max(box[[0,2,4,6,8,10,12,14]])
                        ymin = min(box[[1,3,5,7,9,11,13,15]])
                        ymax = max(box[[1,3,5,7,9,11,13,15]])
                        occupied_x.append([xmin,xmax])
                        occupied_y.append([ymin,ymax])
                    
                    attempts = 0
                    good = False
                    while not good and attempts < 10:
                        good = True
                        xsplit = np.random.randint(0,im.size[0])
                        for rang in occupied_x:
                            if xsplit > rang[0] and xsplit < rang[1]:
                                good = False
                                attempts += 1
                                break
                        if good:
                            break
                    
                    attempts = 0
                    good = False
                    while not good and attempts < 10:
                        good = True
                        ysplit = np.random.randint(0,im.size[1])
                        for rang in occupied_y:
                            if ysplit > rang[0] and ysplit < rang[1]:
                                good = False
                                attempts += 1
                                break
                        if good:
                            break
                    
                    #print(xsplit,ysplit)
                    
                    im11 = im_t[:,:ysplit,:xsplit]
                    im12 = im_t[:,ysplit:,:xsplit]
                    im21 = im_t[:,:ysplit,xsplit:]
                    im22 = im_t[:,ysplit:,xsplit:]
                
                    if TILE > 0.25 and TILE < 0.5:
                        im_t = torch.cat((torch.cat((im21,im22),dim = 1),torch.cat((im11,im12),dim = 1)),dim = 2)
                    elif TILE > 0.5 and TILE < 0.75: 
                        im_t = torch.cat((torch.cat((im22,im21),dim = 1),torch.cat((im12,im11),dim = 1)),dim = 2)
                    elif TILE > 0.75:
                        im_t = torch.cat((torch.cat((im12,im11),dim = 1),torch.cat((im22,im21),dim = 1)),dim = 2)
                    
                    if TILE > 0.25 and TILE < 0.75:
                        for idx in range(0,len(y)):
                            if occupied_x[idx][0] > xsplit:
                                y[idx,[0,2,4,6,8,10,12,14,16,18]] = y[idx,[0,2,4,6,8,10,12,14,16,18]] - xsplit
                            else:
                                y[idx,[0,2,4,6,8,10,12,14,16,18]] = y[idx,[0,2,4,6,8,10,12,14,16,18]] + (im_t.shape[2] - xsplit)
                                
                    if TILE > 0.5:
                         for idx in range(0,len(y)):
                            if occupied_y[idx][0] > ysplit:
                                y[idx,[1,3,5,7,9,11,13,15,17,19]] = y[idx,[1,3,5,7,9,11,13,15,17,19]] - ysplit
                            else:
                                y[idx,[1,3,5,7,9,11,13,15,17,19]] = y[idx,[1,3,5,7,9,11,13,15,17,19]] + (im_t.shape[1] - ysplit)
                    
            #append vp (actually we only need one copy but for simplicity append it to every label)
            # vps = vps.unsqueeze(0).repeat(len(y),1).float()
            # y = y.float()
            # y = torch.cat((y,vps),dim = 1)
        
        
        
        elif self.CROP > 0:
            
            classes = y[:,21].clone() 
            ids = y[:,20].clone()
            
            # use one object to define center
            if y[0,0] != -1:
                
                if crop_index is None:
                    idx = np.random.randint(len(y))
                else:
                    idx = crop_index
    
                box = y[idx]
                centx = (box[16] + box[18])/2.0
                centy = (box[17] + box[19])/2.0
                noise = np.random.normal(0,20,size = 2)
                centx += noise[0]
                centy += noise[1]
                
                size = max(box[19]-box[17],box[18] - box[16])
                size_noise = max( -(size*1/4) , np.random.normal(size*1/4,size/4))
                size += size_noise
                
                if size < 50:
                    size = 50
            else:
                size = max(50,np.random.normal(300,25))
                centx = np.random.randint(100,1000)
                centy = np.random.randint(100,1000)
            try:
                minx = int(centx - size/2)
                miny = int(centy - size/2)
                maxx = int(centx + size/2)
                maxy = int(centy + size/2)
            
            except TypeError:
                print(centx,centy,size)
            
            try:
                im_crop = F.crop(im,miny,minx,maxy-miny,maxx-minx)
                
                if self.multiple_frames:
                    prev_im_crop = F.crop(prev_im,miny,minx,maxy-miny,maxx-minx)
                    
            except:
                print (miny,minx,maxy,maxx,size,centx,centy)
                im_crop = im.copy()
                y = torch.zeros([1,22]) -1
            del im ,prev_im
            
            if im_crop.size[0] == 0 or im_crop.size[1] == 0:
                print("Oh no! {} {} {}".format(centx,centy,size))
                raise Exception
                
            # shift labels if there is at least one object
       
            if y[0,0] != -1:
                
                y[:,::2] -= minx
                y[:,1::2] -= miny
                
    
            crop_size = im_crop.size
            im_crop = F.resize(im_crop, (self.CROP,self.CROP))
            
            if self.multiple_frames:
                prev_im_crop = F.resize(prev_im_crop,(self.CROP,self.CROP))
    
            y[:,::2]  *=  self.CROP/crop_size[0]
            y[:,1::2] *=  self.CROP/crop_size[1]
        
            
            # remove all labels that aren't in crop
            if torch.sum(y) != 0:
                keepers = []
                for i,item in enumerate(y):
                    if item[16] < self.CROP-15 and item[18] > 0+15 and item[17] < self.CROP-15 and item[19] > 0+15:
                        keepers.append(i)
                y = y[keepers]
                classes = classes[keepers]
                ids = ids[keepers]
            
            if len(y) == 0:
                y = torch.zeros([1,22]) -1
                
            else:
                y[:,21] = classes
                y[:,20] = ids
                
                
            # finally convert crop to tensor
            im_t = self.im_tf(im_crop)
            mag = 1 * torch.rand(1).item()
            bias = 3* (torch.rand(1).item() - 0.5)
            randoms = (torch.rand(im_t.shape) * mag - 0.5*mag) + bias
            im_t += randoms
            
            if self.multiple_frames:
                prev_im_t = self.im_tf(prev_im_crop)
                randoms = (torch.rand(im_t.shape) * mag - 0.5*mag) + bias
                prev_im_t += randoms
                
                im_t = torch.cat((im_t,prev_im_t),dim = 0)

            if self.mode =="train":
                OCCLUDE = np.random.rand()
                if OCCLUDE > 0.95:
                    # roughly occlude bottom, left or right third of image
                    yo_min = np.random.randint(im_t.shape[2]/3,im_t.shape[2])
                    yo_max = im_t.shape[2]
                    xo_min = np.random.randint(0,im_t.shape[1]/3)
                    xo_max = np.random.randint(im_t.shape[1]*2/3,im_t.shape[1])
                    region = torch.tensor([xo_min,yo_min,xo_max,yo_max]).int()
                    
                    r =  torch.normal(0.485,0.229,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
                    g =  torch.normal(0.456,0.224,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
                    b =  torch.normal(0.406,0.225,[int(region[3])-int(region[1]),int(region[2])-int(region[0])])
                    rgb = torch.stack([r,g,b])
                    im_t[:3,int(region[1]):int(region[3]),int(region[0]):int(region[2])] = rgb 
                
                    if self.multiple_frames:
                        im_t[3:,int(region[1]):int(region[3]),int(region[0]):int(region[2])] = rgb 

            # if self.CROP != 0:
            #     return im_t,y,keepers
            
        return im_t, y
        
    
    def __len__(self):
        return len(self.labels)
    
    def label_to_name(self,num):
        return self.class_dict[num]
        
    
    def show(self,index):
        """ plots all frames in track_idx as video
            SHOW_LABELS - if True, labels are plotted on sequence
            track_idx - int    
        """
        mean = np.array([0.485, 0.456, 0.406])
        stddev = np.array([0.229, 0.224, 0.225])
        cls_idx = 21
        id_idx = 20
        im,label = self[index]
        
        im2 = torch.zeros(3,im.shape[1],im.shape[2])
        # im2[:,:500,:] = im[:3,:500,:]
        # im2[:,500:,:] = im[3:,500:,:]
        im2[:,:,:] = im[:3,:,:]
        im = im2
        
        im = self.denorm(im)
        cv_im = np.array(im) 
        cv_im = np.clip(cv_im, 0, 1)
        
        # Convert RGB to BGR 
        cv_im = cv_im[::-1, :, :]         
        
        cv_im = np.moveaxis(cv_im,[0,1,2],[2,0,1])

        cv_im = cv_im.copy()

        # class_colors = [
        #     (255,150,0),
        #     (255,100,0),
        #     (255,50,0),
        #     (0,255,150),
        #     (0,255,100),
        #     (0,255,50),
        #     (0,100,255),
        #     (0,50,255),
        #     (255,150,0),
        #     (255,100,0),
        #     (255,50,0),
        #     (0,255,150),
        #     (0,255,100),
        #     (0,255,50),
        #     (0,100,255),
        #     (0,50,255),
        #     (200,200,200) #ignored regions
        #     ]
    
        class_colors = [
            (0,255,0),
            (255,0,0),
            (0,0,255),
            (255,255,0),
            (255,0,255),
            (0,255,255),
            (255,100,0),
            (255,50,0),
            (0,255,150),
            (0,255,100),
            (0,255,50)]
        
        # if self.label_format == "tailed_footprint":
        #     for bbox in label:
        #         thickness = 2
        #         bbox = bbox.int().data.numpy()
        #         cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), class_colors[bbox[-1]], thickness)
        #         cent_x = int((bbox[0] + bbox[2] + bbox[4] + bbox[6])/4.0)
        #         cent_y = int((bbox[1] + bbox[3] + bbox[5] + bbox[7])/4.0)
                
        #         cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[2]+bbox[8],bbox[3]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[0]+bbox[8],bbox[1]+bbox[9]),(bbox[4]+bbox[8],bbox[5]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[2]+bbox[8],bbox[3]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
        #         cv2.line(cv_im,(bbox[4]+bbox[8],bbox[5]+bbox[9]),(bbox[6]+bbox[8],bbox[7]+bbox[9]), class_colors[bbox[-1]], thickness)
                
        #         plot_text(cv_im,(bbox[0],bbox[1]),bbox[-1],0,class_colors,self.classes)
                
        if self.label_format == "8_corners":
            for bbox in label:
                thickness = 1
                bbox = bbox.int().data.numpy()
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), (0,255,0), thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), (255,0,0), thickness)
                
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), class_colors[bbox[cls_idx]], thickness)
                
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), class_colors[bbox[cls_idx]], thickness)
                cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), (0,0,255), thickness)

                cv2.rectangle(cv_im, (bbox[16],bbox[17]),(bbox[18],bbox[19]),class_colors[bbox[cls_idx]],thickness)
                
                
        
                # draw line from center to vp1
                # vp1 = (int(bbox[21]),int(bbox[22]))
                # center = (int((bbox[0] + bbox[2])/2),int((bbox[1] + bbox[3])/2))
                # cv2.line(cv_im,vp1,center, class_colors[bbox[cls_idx]], thickness)
        
        # for region in metadata["ignored_regions"]:
        #     bbox = region.astype(int)
        #     cv2.rectangle(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), class_colors[-1], 1)
       
        #cv_im = cv2.resize(cv_im,(1920,1080))
        cv2.imshow("Frame",cv_im)
        cv2.waitKey(0) 
        
def collate(inputs):
        """
        Receives list of tuples and returns a tensor for each item in tuple, except metadata
        which is returned as a single list
        """
        im = [] # in this dataset, always [3 x W x H]
        label = [] # variable length
        max_labels = 0
        
        for batch_item in inputs:
            im.append(batch_item[0])
            label.append(batch_item[1])
            
            # keep track of image with largest number of annotations
            if len(batch_item[1]) > max_labels:
                max_labels = len(batch_item[1])
            
        # collate images        
        ims = torch.stack(im)
        
        size = len(label[0][0])
        # collate labels
        labels = torch.zeros([len(label),max_labels,size]) - 1
        for idx in range(len(label)):
            num_objs = len(label[idx])
            
            labels[idx,:num_objs,:] = label[idx]
        return ims,labels


if __name__ == "__main__":
    #### Test script here
    
#%%   
    dataset_dir = "/home/worklab/Documents/I24-3D/cache"
    mask_dir = "/home/worklab/Documents/I24-3D/data/mask"
    
    dataset_dir = "/home/worklab/Documents/datasets/I24-3D/cache"
    mask_dir = "/home/worklab/Documents/datasets/I24-3D/data/mask"
    
    test = I24_Dataset(dataset_dir,label_format = "8_corners",mode = "train", CROP = 0, multiple_frames=False,mask_dir = mask_dir,random_partition = True)
    
    
    
    for i in range(100):
        idx = np.random.randint(0,len(test))
        test.show(idx)
    #cv2.destroyAllWindows()
