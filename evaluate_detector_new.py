#%% Imports
import warnings; warnings.filterwarnings("ignore")

import os ,sys
import numpy as np
import random 
import cv2
import time
import torch
from torch.utils import data
from torch import optim
import re
import collections
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from datetime import datetime
import _pickle as pickle

#from detection_dataset_3D_multitask import Detection_Dataset, collate
from multitask_dataset import I24_Dataset,collate

random.seed(0)
torch.manual_seed(0)

import matplotlib.pyplot as plt 

#%% utility

def state_to_footprint(boxes):
    d = boxes.shape[0]
    intermediate_boxes = torch.zeros([d,4,2], device = boxes.device)
    intermediate_boxes[:,0,0] = boxes[:,0] 
    intermediate_boxes[:,0,1] = boxes[:,1] - boxes[:,3]/2.0
    intermediate_boxes[:,1,0] = boxes[:,0] 
    intermediate_boxes[:,1,1] = boxes[:,1] + boxes[:,3]/2.0
    
    intermediate_boxes[:,2,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
    intermediate_boxes[:,2,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] - boxes[:,3]/2.0
    intermediate_boxes[:,3,0] = boxes[:,0] + boxes[:,2]*boxes[:,5]
    intermediate_boxes[:,3,1] = boxes[:,1] + boxes[:,2]*boxes[:,5] + boxes[:,3]/2.0

    boxes_new = torch.zeros([boxes.shape[0],4],device = boxes.device)
    boxes_new[:,0] = torch.min(intermediate_boxes[:,0:4,0],dim = 1)[0]
    boxes_new[:,2] = torch.max(intermediate_boxes[:,0:4,0],dim = 1)[0]
    boxes_new[:,1] = torch.min(intermediate_boxes[:,0:4,1],dim = 1)[0]
    boxes_new[:,3] = torch.max(intermediate_boxes[:,0:4,1],dim = 1)[0]
    return boxes_new


def test_detector_video(retinanet,video_path,dataset,break_after = 12000,detect_every = 3):
    """
    Use current detector on frames from specified video
    """
    retinanet.training = False
    retinanet.eval()
    cap = cv2.VideoCapture(video_path)
    all_data = []
    
    for i in range(break_after):
        if i % detect_every == 0:
            print("On frame {} of {}".format(i,break_after))
            
            ret,frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame,(1920,1080))
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
            frame = F.to_tensor(frame)
            #frame = frame.permute((2,0,1))
            
            frame = F.normalize(frame,mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    
            im = frame.to(device).unsqueeze(0).float()
            im = torch.cat((im,im),dim = 1).half()
            
            
            with torch.no_grad():
                scores,labels,boxes,_ = retinanet(im)
             
            if len(boxes) > 0:
                keep = torch.where(scores > 0.5,1,0).nonzero().squeeze(1)
                boxes = boxes[keep,:].cpu()
                scores = scores[keep].cpu()
                labels = labels[keep].cpu().data.numpy()
            im = dataset.denorm(im[0,:3])
            cv_im = np.array(im.cpu()) 
            cv_im = np.clip(cv_im, 0, 1)
        
            # Convert RGB to BGR 
            cv_im = cv_im[::-1, :, :]  
            cv_im = cv_im.transpose((1,2,0))*255 
            cv_im = cv_im.astype(np.uint8)
            cv_im = cv_im.copy()
        
            thickness = 1
            all_data.append([i,boxes,scores,labels])
            for bidx,bbox in enumerate(boxes):
                thickness = 2
                
                color = dataset.class_colors[labels[bidx]]
                if labels[bidx] not in [4,5,6]:
                    color = (200,200,200)
                    thickness = 1
                
                bbox = bbox.int().data.cpu().numpy()
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), color, thickness)
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), color, thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), color, thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), color, thickness)
                
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), color, thickness)
                cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), color, thickness)
                cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), color, thickness)
                cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), color, thickness)
                
                cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), color, thickness)
                cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), color, thickness)
                cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), color, thickness)
                cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), color, thickness)
                
                # put text label at top left
                top = np.min(bbox[::2])
                left = np.min(bbox[1::2])
                cv2.putText(cv_im,dataset.classes[labels[bidx]],(int(top),int(left)),cv2.FONT_HERSHEY_SIMPLEX,1,color,thickness)
                
            cv2.imshow("Frame",cv_im)
            #cv2.imwrite("/home/worklab/Documents/i24/I24-3D-dataset/temp_vid_out/{}.png".format(str(i).zfill(5)), cv_im)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
    
    else:
        cap.grab()
            
    cv2.destroyAllWindows()
    cap.release()
    
    with open("newtruck_detections_P27C01.cpkl","wb") as f:
        pickle.dump(all_data,f)

def ravel_to_npy():
    with open("newtruck_detections_P27C01.cpkl","rb") as f:
        all_data = pickle.load(f)
        
        
        # load v3 homography
        from i24_rcs import I24_RCS
        rcs = I24_RCS("/home/worklab/Documents/i24/I24-3D-dataset/hg_67e62dfd1ffd2fe3d61ee2d0.cpkl",default = "reference",downsample = 2)
        
        
        # test = torch.arange(0,50000,step = 100)
        # test2 = torch.arange(-120,120,step = 12)
        # test = test.unsqueeze(1).expand(test.shape[0],test2.shape[0])
        # test2 = test2.unsqueeze(0).expand(test.shape)
        # test = test.reshape(-1)
        # test2 = test2.reshape(-1)
        # points = torch.stack([test,test2,test2*0,test2*0,test2*0,torch.sign(test2)]).transpose(1,0).contiguous()
        # space_pts = rcs.state_to_space(points)
        
        # plt.scatter(space_pts[:,0,0],space_pts[:,0,1],color = "k")
        det = np.empty([0,8])
        
        # map each to state form
        for frame_data in all_data:
            fidx = frame_data[0]
            boxes = frame_data[1]
            scores = frame_data[2]
            labels = torch.from_numpy(frame_data[3])
        
            
            # convert boxes to [n_deteections,8,2]
            
            detections = boxes.view(-1,10,2)
            boxes = detections[:,:8,:] # drop 2D boxes
            
            #boxes = boxes.view(-1,8,2) # this step may be wrong
            boxes_state = rcs.im_to_state(boxes, name = ["P27C01" for _ in boxes],classes = labels)
            #plt.scatter(boxes_state[:,0,0],boxes_state[:,0,1])
            #add time
            time = torch.zeros([boxes.shape[0],1]) + fidx* 0.0333
            
            # add conf and class
            boxes_final = torch.cat((time,boxes_state[:,:5],labels.unsqueeze(1),scores.unsqueeze(1)),dim = 1).data.numpy()
            
        
            det = np.concatenate((det,boxes_final),axis = 0)
            print(fidx)
            
            #plt.scatter(det[:,1],det[:,2])

        # time x y l w h class confidenc,e original id clustered id
        np.save("example_detections_P27C01.npy",det)
        
        print("X-range: {} {}".format(np.min(det[:,1]),np.max(det[:,1])))

def smooth_tracklets(file):
    def traj_to_tensor(traj,cls = -1):
        t = torch.zeros(len(traj["x_position"]),7)
        t[:,0] = torch.tensor(traj["timestamp"])
        t[:,1] = torch.tensor(traj["x_position"])
        t[:,2] = torch.tensor(traj["y_position"])
        t[:,3] = traj["length"]
        t[:,4] = traj["width"]
        t[:,5] = traj["height"]
        t[:,6] = cls
        return t
    
    from utils_opt import resample,opt2_l1_constr
    lam1_x= 3e-1
    lam2_x= 0
    lam3_x= 1e-7
    
    lam1_y= 0
    lam2_y= 0
    lam3_y= 1e-3
    
    
    
    with open(file,"rb") as f:
        tracklets,tracklets_complete = pickle.load(f)
        tracklets += tracklets_complete
    
    
    
    
    smoothed = []
    for tidx in range(len(tracklets)):
    # iterate over trajectories
        print("On tracklet {}".format(tidx))
        traj = tracklets[tidx]
        counts = np.bincount(traj[:,6],minlength = 8)
        cls = np.argmax(counts)
        # wrangle form for resampling
        car = {"timestamp" : traj[:,0].data.numpy(),
               "x_position": traj[:,1].data.numpy(),
               "y_position": traj[:,2].data.numpy(),
               "direction" : torch.sign(traj[0,2]).item(),
               "length":torch.mean(traj[:,3]),
               "width" :torch.mean(traj[:,4]),
               "height":torch.mean(traj[:,5])
               }
        try:
            re_car = resample(car,dt = 0.04,fillnan = True)
            smooth_car = opt2_l1_constr(re_car.copy(), lam2_x, lam2_y, lam3_x, lam3_y, lam1_x, lam1_y)
            # re_car = traj_to_tensor(re_car)
            smooth_car = traj_to_tensor(smooth_car,cls = cls)
            smoothed.append(smooth_car)
        except ZeroDivisionError:
            print("Error zero division")
    
    out_file = file.split(".cpkl")[0] + "_smooth.cpkl"
    with open(out_file,"wb") as f:
        pickle.dump(smoothed,f)

def plot_smoothed_video(video_path,traj_path,dataset):
    """
    Use current detector on frames from specified video
    """
    
    # load v3 homography
    from i24_rcs import I24_RCS
    rcs = I24_RCS("/home/worklab/Documents/i24/I24-3D-dataset/hg_67e62dfd1ffd2fe3d61ee2d0.cpkl",default = "reference",downsample = 2)
    
    
    retinanet.training = False
    retinanet.eval()
    cap = cv2.VideoCapture(video_path)
    
    with open(traj_path,"rb") as f:
        tracklets = pickle.load(f)
    
    
    
    frame_idx = 0
    while True:
            
            ret,frame = cap.read()
            if not ret:
                break
            
            #frame = cv2.resize(frame,(1920,1080))
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    
            ts = frame_idx * 0.1
        
            # get set of boxes
            boxes = []
            classes = []  
            trucks = []
            truck_classes = []
            for traj_idx,t in enumerate(tracklets):
                if t[0,0] < ts and t[-1,0] > ts:
                    tidx = 0
                    while t[tidx,0] < ts:
                        tidx += 1
                    
                    boxes.append(t[tidx,1:6])
                    classes.append(t[tidx,6].item())
                    if t[tidx,6].item() in [5,6,7]:
                        trucks.append(t[tidx,1:6])
                        truck_classes.append(t[tidx,6].item())
        
            #stack
            if len(boxes) > 0:
                # boxes is [n_objects,6] tensor - xc,y,l,w,h,direction (sign(y))
                boxes = torch.stack(boxes)
                boxes = torch.cat((boxes,torch.sign(boxes[:,1]).unsqueeze(1)),dim = 1)
                rcs.plot_state_boxes(frame,boxes,name = ["P27C01" for _ in classes])
                rcs.state_to_im(boxes,name =  ["P27C01" for _ in classes],times = None)
                
            if len(trucks) > 0:
                boxes = torch.stack(trucks)
                boxes = torch.cat((boxes,torch.sign(boxes[:,1]).unsqueeze(1)),dim = 1)
                rcs.plot_state_boxes(frame,boxes,name = ["P27C01" for _ in truck_classes],thickness = 2,color = (100,100,250),labels = [dataset.classes[i] for i in truck_classes])

            # all_data.append([i,boxes,scores,labels])
            # for bidx,bbox in enumerate(boxes):
            #     thickness = 2
                
            #     color = dataset.class_colors[labels[bidx]]
            #     if labels[bidx] not in [4,5,6]:
            #         color = (200,200,200)
            #         thickness = 1
                
            #     bbox = bbox.int().data.cpu().numpy()
            #     cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[2],bbox[3]), color, thickness)
            #     cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[4],bbox[5]), color, thickness)
            #     cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[6],bbox[7]), color, thickness)
            #     cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[6],bbox[7]), color, thickness)
                
            #     cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[10],bbox[11]), color, thickness)
            #     cv2.line(cv_im,(bbox[8],bbox[9]),(bbox[12],bbox[13]), color, thickness)
            #     cv2.line(cv_im,(bbox[10],bbox[11]),(bbox[14],bbox[15]), color, thickness)
            #     cv2.line(cv_im,(bbox[12],bbox[13]),(bbox[14],bbox[15]), color, thickness)
                
            #     cv2.line(cv_im,(bbox[0],bbox[1]),(bbox[8],bbox[9]), color, thickness)
            #     cv2.line(cv_im,(bbox[2],bbox[3]),(bbox[10],bbox[11]), color, thickness)
            #     cv2.line(cv_im,(bbox[4],bbox[5]),(bbox[12],bbox[13]), color, thickness)
            #     cv2.line(cv_im,(bbox[6],bbox[7]),(bbox[14],bbox[15]), color, thickness)
                
            #     # put text label at top left
            #     top = np.min(bbox[::2])
            #     left = np.min(bbox[1::2])
            #     cv2.putText(cv_im,dataset.classes[labels[bidx]],(int(top),int(left)),cv2.FONT_HERSHEY_SIMPLEX,1,color,thickness)
                
            cv2.imshow("Frame",frame)
            #cv2.imwrite("/home/worklab/Documents/i24/I24-3D-dataset/temp_vid_out/{}.png".format(str(frame_idx).zfill(5)), frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            
            frame_idx += 1
            
    
    else:
        cap.grab()
            
    cv2.destroyAllWindows()
    cap.release()

def md_iou(a,b):
    """
    a,b - [batch_size ,num_anchors, 4]
    """
    
    area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
    area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
    
    minx = torch.max(a[:,:,0], b[:,:,0])
    maxx = torch.min(a[:,:,2], b[:,:,2])
    miny = torch.max(a[:,:,1], b[:,:,1])
    maxy = torch.min(a[:,:,3], b[:,:,3])
    zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection
    iou = torch.div(intersection,union)
    
    #print("MD iou: {}".format(iou.max(dim = 1)[0].mean()))
    return iou


def get_metrics(confusion_matrix):
    
    #plot confusion matrix
    sums = np.sum(confusion_matrix,axis= 0)
    sumss = sums[:,np.newaxis]
    sumss = np.repeat(sumss,8,1)#.transpose()
    sumss = np.transpose(sumss)
    percentages = np.round(confusion_matrix/sumss * 100)
    
    fig, ax = plt.subplots(figsize = (10,10))
    im = ax.imshow(percentages,cmap = "YlGn")
    
    classes = [val_data.classes[i] for i in range (0,8)]
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_ylim(len(classes)-0.5, -0.5)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center",
         rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, confusion_matrix[i, j],
                       ha="center", va="bottom", color="k",fontsize = 20)
            text = ax.text(j, i, str(percentages[i, j])+"%",
                       ha="center", va="top", color="k",fontsize = 14)
    
    ax.set_title("Test Data Confusion Matrix",fontsize = 20)
    ax.set_xlabel("Actual",fontsize = 20)
    ax.set_ylabel("Predicted",fontsize = 20)
    plt.show()
    
    # get overall accuracy
    correct = sum([confusion_matrix[i,i] for i in range(0,8)])
    total = np.sum(confusion_matrix)
    accuracy = correct/total
    print("Test accuracy: {}%".format(accuracy*100))

    # get per-class recall (correct per class/ number of items in this class)    
    correct_per_class = np.array([confusion_matrix[i,i] for i in range(0,8)])
    recall = correct_per_class/sums
    
    total_preds_per_class = np.sum(confusion_matrix,axis= 1)
    precision = correct_per_class/total_preds_per_class
    
    return accuracy,recall,precision




#%% Specify detector here
if True: # Resnet34 with multiple frames and embedding
    # add relevant packages and directories to path
    detector_path = os.path.join(os.getcwd(),"pytorch_retinanet_detector_multi_frame")
    sys.path.insert(0,detector_path)
    from pytorch_retinanet_detector_multi_frame.retinanet.model import resnet34
    
    # load detector
    depth = 34
    num_classes = 8
    checkpoint_file = "/home/worklab/Documents/i24/I24-3D-dataset/cp/truck_running_checkpoint_retrain.pt" # highly trained
    #checkpoint_file = "/home/worklab/Documents/i24/I24-3D-dataset/cp/truck_running_checkpoint_save.pt" #old
    
    # Create the model
    if depth == 18:
        retinanet = resnet18(num_classes=num_classes, pretrained=True)
    elif depth == 34:
        retinanet = resnet34(num_classes=num_classes, pretrained=True)
    elif depth == 50:
        retinanet = resnet50(num_classes=num_classes, pretrained=True)
    elif depth == 101:
        retinanet = resnet101(num_classes=num_classes, pretrained=True)
    elif depth == 152:
        retinanet = resnet152(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
     
    conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv1.weight.data[:,:3,:,:] = retinanet.conv1.weight.data.clone()
    conv1.weight.data[:,3:,:,:] = retinanet.conv1.weight.data.clone()
    retinanet.conv1 = conv1
    
    # load checkpoint if necessary
    try:
        if checkpoint_file is not None:
            retinanet.load_state_dict(torch.load(checkpoint_file).state_dict())
    except:
        retinanet.load_state_dict(torch.load(checkpoint_file))
    
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    retinanet = retinanet.to(device)
    
    
    # eval mode
    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn()

    retinanet = retinanet.half()
    for layer in retinanet.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.float()



#%% Get datatset
data_dir = "/home/derek/Data/cv/internal_datasets/test_cache"
data_dir = "/home/worklab/Documents/datasets/I24-3D/cache"
val_data = I24_Dataset(data_dir,label_format = "8_corners",mode = "test", CROP = 0, multiple_frames=True,mask_dir = None,random_partition = True)


#%% Run evaluation
sigma_min = 0.25
iou_min   = 0.2
nms_iou   = 0.1
n_samples = len(val_data)
CROP      = 0

if True:
    FP = 0
    FN = 0
    TP = 0
    inf_time = 0
    match_IOUs = []
    
    all_embeddings    = []
    all_embedding_ids = []
    
    start = time.time()
    print("Starting test with min confidence {}, min IOU {}, and {} samples".format(sigma_min,iou_min,n_samples))
    
    conf_matrix = np.zeros([8,8])
    
    obj_count = np.zeros(8)
    obj_ious = np.zeros([8,19]) # 0.05 - 1 by 0.05
    
    # load up those sweet sweet homographies
    # load v3 homography
    from i24_rcs import I24_RCS
    rcs1 = I24_RCS("/home/worklab/Documents/datasets/more_3D_frames/hg_67e62dfd1ffd2fe3d61ee2d0.cpkl",default = "reference",downsample = 2)
    #rcs2 = I24_RCS("/home/worklab/Documents/datasets/I24-3D/data/hg/i24_mc3d_rcs.cpkl",aerial_ref_dir="/home/worklab/Documents/datasets/I24-3D/rcs_1/aerial_ref_1",im_ref_dir="/home/worklab/Documents/datasets/I24-3D/rcs_1/cam_ref_1",downsample = 2,default = "reference")
    rcs2 = I24_RCS("/home/worklab/Documents/datasets/I24-3D/data/hg/i24_mc3d_rcs_correspondences.cpkl",downsample = 2, default = "reference")
    
    
    #patching of rcs2
    keylist = list(rcs2.correspondence.keys())
    for key in keylist:
        KEY = key.upper()
        rcs2.correspondence[KEY] = rcs2.correspondence[key]
    
    to_add = []
    for key in rcs2.correspondence.keys():
        if "C4" in key or "C3" in key:
            direction = key.split("_")[1]
            split_key = key.split("_")[0]
            if direction == "WB":
                if split_key + "_EB" not in rcs2.correspondence.keys():
                    to_add.append([split_key + "_EB", rcs2.correspondence[key]])
            elif direction == "EB":
                 if split_key + "_WB" not in rcs2.correspondence.keys():
                     to_add.append([split_key + "_WB", rcs2.correspondence[key]])
    for item in to_add:
        rcs2.correspondence[item[0]] = item[1]
    
    #patching of rcs1
    to_add = []
    for key in rcs1.correspondence.keys():
        if "C0" in key or "C03" in key:
            direction = key.split("_")[1]
            split_key = key.split("_")[0]
            if direction == "WB":
                if split_key + "_EB" not in rcs1.correspondence.keys():
                    to_add.append([split_key + "_EB", rcs1.correspondence[key]])
            elif direction == "EB":
                 if split_key + "_WB" not in rcs1.correspondence.keys():
                     to_add.append([split_key + "_WB", rcs1.correspondence[key]])
    for item in to_add:
        rcs1.correspondence[item[0]] = item[1]
    
    
        
    for idx in range(n_samples):
        # status update
        done_ratio = idx/n_samples + 0.001
        so_far = time.time() - start
        completion = start + so_far/done_ratio
        eta = str(datetime.fromtimestamp(completion)).split(".")[0]
        print("\rOn sample {} of {}, ETA {}".format(idx,n_samples,eta), end = "\r", flush = False)
    
        # get data
        #ridx = np.random.randint(0,len(val_data)-1)
        ridx = idx
        
        
        im,labels = val_data[ridx]
        im = im.unsqueeze(0).to(device).half()
        
        
        # get camera name
        im_path = val_data.data[ridx]
        try: 
            cam_name =  re.search("P\d\dC\d\d",val_data.data[ridx]).group(0)
            rcs = rcs1
        except: 
            cam_name = re.search("p\dc\d",val_data.data[ridx]).group(0)
            rcs = rcs2
        
        # get detector outputs
        with torch.no_grad():
             i_start = time.time()
             scores,classes,boxes,embeddings = retinanet(im)
             #scores,classes,boxes = retinanet(im)
             #embeddings = torch.zeros(10000)
             
             torch.cuda.synchronize()
             inf_time += time.time() - i_start
             
             # boxes is of shape [N,20]
             boxes = boxes.reshape(-1,10,2)
             boxes = boxes[:,:8,:] # drop 2D boxes
             # now of shape [N,8,2]
             
             # convert labels to 2D bbox
             label_ids = labels[:,-2]
             label_classes = labels[:,-1]
             labels = labels.reshape(-1,11,2)
             labels = labels[:,:8,:] # drop 2D boxes
             
             # low confidence filter
             keep = torch.where(scores > sigma_min)
             if len(keep) > 0:
                 scores = scores[keep]
                 classes = classes[keep]
                 boxes  = boxes[keep]
                 embeddings = embeddings[keep]
        
             # im space nms
             if len(boxes) > 0:
                boxes_new = torch.zeros(boxes.shape[0],4,device = device)
                boxes_new[:,0] = torch.min(boxes[:,:,0],dim = 1)[0]
                boxes_new[:,1] = torch.min(boxes[:,:,1],dim = 1)[0]
                boxes_new[:,2] = torch.max(boxes[:,:,0],dim = 1)[0]
                boxes_new[:,3] = torch.max(boxes[:,:,1],dim = 1)[0]
               
                # do nms
                keep = torchvision.ops.nms(boxes_new, scores, iou_threshold = nms_iou)
                scores = scores[keep]
                classes = classes[keep]
                boxes  = boxes[keep]
                embeddings = embeddings[keep]
                
             if False: #2D comparison
                 # convert outputs to 2D bbox
                 boxes_new = torch.zeros(boxes.shape[0],4,device = device)
                 boxes_new[:,0] = torch.min(boxes[:,:,0],dim = 1)[0]
                 boxes_new[:,1] = torch.min(boxes[:,:,1],dim = 1)[0]
                 boxes_new[:,2] = torch.max(boxes[:,:,0],dim = 1)[0]
                 boxes_new[:,3] = torch.max(boxes[:,:,1],dim = 1)[0]
                 boxes = boxes_new
                
                 # do nms
                 keep = torchvision.ops.nms(boxes, scores, iou_threshold = nms_iou)
                 scores = scores[keep]
                 classes = classes[keep]
                 boxes  = boxes[keep]
                 embeddings = embeddings[keep]
                 
                 
                 
                 # # convert labels to 2D bbox
                 # label_ids = labels[:,-2]
                 # label_classes = labels[:,-1]
                 # labels = labels.reshape(-1,11,2)
                 # labels = labels[:,:8,:] # drop 2D boxes
                 
                 labels_new = torch.zeros(labels.shape[0],4)
                 labels_new[:,0] = torch.min(labels[:,:,0],dim = 1)[0]
                 labels_new[:,1] = torch.min(labels[:,:,1],dim = 1)[0]
                 labels_new[:,2] = torch.max(labels[:,:,0],dim = 1)[0]
                 labels_new[:,3] = torch.max(labels[:,:,1],dim = 1)[0]
                 labels = labels_new.to(device)
             
                
             
             else: # 3D footprint comparison
                 if len(scores) > 0:
                     detection_cam_names = [cam_name for _ in scores]
                     # Use the guess and refine method to get box heights
                     boxes = rcs.im_to_state(boxes,name = detection_cam_names,classes = classes)
                     boxes = state_to_footprint(boxes)
                     
                 if len(label_ids) > 0:
                     label_cam_names = [cam_name for _ in label_ids]
                     labels = rcs.im_to_state(labels,name = label_cam_names, classes = label_classes)
                     labels = labels.to(device)
                     labels = state_to_footprint(labels)
                     # convert from state to footprint form
             
                
             # calc iou per pair
             if len(boxes) == 0:
                 a,b = [i for i in range(len(labels))],[-1 for i in range(len(labels))]
                 dist = np.zeros([len(labels),1])
             elif len(labels) == 0:
                 a,b = [],[]
                 dist = None
             else:  
                 f = labels.shape[0]
                 s = boxes.shape[0]
                 second = boxes.unsqueeze(0).repeat(f,1,1).double()
                 first  = labels.unsqueeze(1).repeat(1,s,1).double()
                 dist   = md_iou(first,second).cpu()
       
            
        
                 # match detections to labels
                 try:
                     a, b = linear_sum_assignment(dist.data.cpu().numpy(),maximize = True) 
                 except ValueError:
                      print("ERROR!!!")
              
             # there is one entry in a per label
             # there is one entry in b per label
             # a indexes labels
             # b indexes boxes
             
             # # convert into expected form
             # matchings = np.zeros(s)-1
             # for idx in range(0,len(b)):
             #      matchings[b[idx]] = a[idx]
             # matchings = np.ndarray.astype(matchings,int)
              
             # # remove any matches too far away
             # # TODO - Vectorize this
             # for i in range(len(matchings)):
             #     if matchings[i] != -1 and  dist[matchings[i],i] < iou_min:
             #         matchings[i] = -1    
                     
             
            # aggregate score metrics
             this_tp = 0
             for i in range(len(a)):
                 label_cls = int(label_classes[a[i]].item())
                 iou = dist[a[i],b[i]]
                 obj_count[label_cls] += 1
                 obj_ious[label_cls,:int(iou/5*100)] += 1
                 
                 if b[i] >= 0 and a[i] >= 0 and iou > iou_min:
                     match_IOUs.append(iou)
                     this_tp += 1
                     
                     
                     # get class of label and pred
                     pred_cls =  int(classes[b[i]].item())
                     # row is true,column is prediction
                     conf_matrix[label_cls,pred_cls] += 1
                     
                     all_embedding_ids.append(label_ids[a[i]])
                     all_embeddings.append(embeddings[b[i]])
             TP += this_tp
             FP += (len(boxes)  - this_tp)
             FN += (len(labels) - this_tp)
             
             try:
                 assert (len(boxes)  - this_tp) >= 0 and (len(labels) - this_tp) >= 0, "Assert error"
             except AssertionError:
                print(1)
    
    # summary metrics
    precision = TP/(TP + FP)
    recall    = TP/(TP + FN)
    F1        = 2*precision*recall/(precision + recall)
    avg_IOU   = sum(match_IOUs) / len(match_IOUs)
    
    
    
    
    
    if False:
    
        id_sub = torch.stack(all_embedding_ids)
        nid = id_sub.shape[0]
        
        # emb_sub.shape = [nid,embedding]
        emb_sub = torch.stack(all_embeddings).double()
        
        # same_id[i,j] = 1 if id_sub i and j are the same id, else 0
        id_match = id_sub.unsqueeze(1).expand(nid,nid)
        same_id = (id_match == id_match.transpose(0,1)).to(device).int()
        
        # [nid,nid,embedding_size]
        emb_exp = emb_sub.unsqueeze(0).expand(nid,nid,emb_sub.shape[1])
        
        similarity = torch.cosine_similarity(emb_exp,emb_exp.transpose(0,1),dim = 2)
        similarity *= (1-torch.eye(nid,device = device).int())
        
        
        best_match_is_correct = 0
        best_match_is_incorrect = 0
        best_score = []
        highest_score_no_match = []
        for i in range(nid):
            max_sim = similarity[i].max()
            max_idx = similarity[i].argmax()
            
            if same_id[i,max_idx] == 1:
                best_match_is_correct += 1
                best_score.append(max_sim)
                
            elif same_id[i].sum() > 0:
                best_match_is_incorrect += 1
                highest_score_no_match.append(max_sim)
            
    
    # best_ratio = 0
    # best_cutoff = 0
    # best_recall = 0
    # best_precision = 0
    # for cutoff in  np.linspace(-1,1,200):
    #     hits = (similarity > cutoff).int()
    #     hits *= (1-torch.eye(nid,device = device).int())
    #     TP = (hits * same_id).sum()
    #     FP = (hits * (1-same_id)).sum()
         
        
    #     misses = (similarity < cutoff).int()
    #     misses *= (1-torch.eye(nid,device = device).int())
    #     TN = (misses * (1-same_id)).sum()
    #     FN = (misses * same_id).sum()
    
        
    #     hit_recall     = TP / (TP + FN)
    #     hit_precision  = TP / (TP + FP)
    #     miss_recall    = TN / (TN + FP)
        
    #     total_score = 2*hit_recall*hit_precision/(hit_recall + hit_precision)
    #     #print(cutoff, hit_recall, hit_precision)
    #     if total_score > best_ratio:
    #         best_ratio     = total_score
    #         best_cutoff    = cutoff
    #         best_recall    = hit_recall
    #         best_precision = hit_precision
            
    
    
    # append class errors to heatmap
    
    print("\nPrecision: {:.3f}".format(precision))
    print("Recall: {:.3f}".format(recall))
    print("F1: {:.3f}".format(F1))
    print("Avg. IOU: {:.3f}".format(avg_IOU))
    print("Inference Time (ms): {:.0f}".format(inf_time/n_samples * 1000))
    
    # plot confusion matrix
    get_metrics(conf_matrix)

    # plot iou curve
    plt.figure(figsize = (10,10))
    
    for i in range(len(obj_count)):
        iou_curve = obj_ious[i] / obj_count[i]
        xticks = [j*0.05 for j in range(len(iou_curve))]
        plt.plot(xticks,iou_curve,":")
    
   
    
    iou_total = obj_ious.sum(axis = 0) / obj_count.sum()    
    plt.plot(xticks,iou_total,"k")

    plt.legend(["{} - ({}) samples".format(val_data.classes[i],int(obj_count[i])) for i in range(len(obj_count))] + ["Mean- ({}) samples".format(int(obj_count.sum()))])
    plt.xlabel("Min required IOU")
    plt.ylabel("Recall")
    plt.title("Recall at Varying IOU Thresholds by Class, New")
    


#%%
# load a video and detect in it
if False:

    video_path = "/home/worklab/Desktop/temp_vid/P22C01_1744718342.47871.mkv"
    test_detector_video(retinanet,video_path,val_data,break_after = 8000)
    
if False:
    ravel_to_npy()

if False:
    tracklet_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_0/tracklets_0_600_9200_9600.cpkl"
    tracklet_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_1/tracklets_0_600_22800_23600.cpkl"
    tracklet_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_1/tracklets_0_1000_23300_24000.cpkl"        
    smooth_tracklets(tracklet_path)
    
if False:
    video_path  = "/home/worklab/Desktop/temp_vid/P01C02_1743079738.921835.mkv"
    traj_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_0/tracklets_0_600_9200_9600_smooth.cpkl"
    
    video_path = "/home/worklab/Desktop/temp_vid/P26C01_1743424742.252281.mkv"
    traj_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_1/tracklets_0_600_22800_23600_smooth.cpkl"    
    
    video_path = "/home/worklab/Desktop/temp_vid/P27C01_1743162549.046082.mkv"
    traj_path = "/home/worklab/Documents/i24/cluster-track-dev/data/truck_1/tracklets_0_1000_23300_24000_smooth.cpkl"       
    plot_smoothed_video(video_path,traj_path,val_data)
        