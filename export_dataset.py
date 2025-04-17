from scene import Scene

import numpy as np
import torch
import _pickle as pickle
import copy
import cv2
import os
import time
import gc

start_time = time.time()
try:
    with open("/home/worklab/Documents/datasets/I24-3D/cache/data_summary.cpkl","rb") as f:
        all_data = pickle.load(f)
except:
        all_data = []

def create_summary(directories,out_dir):
    all_data = []
    for directory in directories:
        
        im_dir    = directory[0]
        label_dir = directory[1]
        
        ims = os.listdir(im_dir)
        labels = os.listdir(label_dir)
          
        if os.path.isdir(os.path.join(im_dir,ims[0])): 
            for cam in ims:
                im_dir_new = os.path.join(im_dir,cam)
                label_dir_new = os.path.join(label_dir,cam)
                
                ims_new = os.listdir(im_dir_new)
                labels_new = os.listdir(label_dir_new)
                
                for im in ims_new:
                    label = im.split(".png")[0] + ".cpkl"
                    if label in labels_new:
                        all_data.append([os.path.join(im_dir_new,im),os.path.join(label_dir_new,label)])
            
        else:
            for im in ims:
                label = im.split(".png")[0] + ".cpkl"
                if label in labels:
                    all_data.append([os.path.join(im_dir,im),os.path.join(label_dir,label)])
    
    save_label_path = "{}/data_summary.cpkl".format(out_dir)
    with open(save_label_path,"wb") as f:
        pickle.dump(all_data,f)
    print("Saved dataset summary with {} items at {}".format(len(all_data),save_label_path))

def cache_frames(ann,output_directory, start_frame = 0,how_many = 8,append_extra = False):
    
    if not append_extra:
        # for each frame idx
        for f_idx in np.arange(start_frame,ann.last_frame):
            
            if len(ann.data[f_idx]) == 0: # past last frame
                break
            
            # for each camera
            if f_idx % how_many == 0: # only cache every 20th frame
                
                for c_idx in range(len(ann.cameras)):
                    cam_frame_annotations = []
                    
                    camera = ann.cameras[c_idx]
                    cam_name = ann.cameras[c_idx].name
                    
                    # get set of objects labeled within that camera
                    ts_data = list(ann.data[ann.frame_idx].values())
                    ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
                    
                    if len(ts_data) == 0:
                        continue
                    ids = [item["id"] for item in ts_data]
                    if len(ts_data) > 0:
                        boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
                        
                        # convert into image space
                        im_boxes = ann.hg.state_to_im(boxes,name = camera.name)
                         
                    for i in range (len(ts_data)):
                        # for each object, append the image-space box to the ts_data annotation
                        ts_data[i]["im_box"] = im_boxes[i]
                        ts_data[i]["box"] = boxes[i]
                        ts_data[i]["frame_idx"] = f_idx
                        if "gen" not in ts_data[i].keys(): ts_data[i]["gen"] = "Manual"
                        cam_frame_annotations.append(ts_data[i])
                    
                    # generate image path
                    im_directory = "{}/im".format(output_directory)
                    if not os.path.exists(im_directory):
                            os.mkdir(im_directory)
                    scene_directory = "{}/im/scene_{}".format(output_directory,ann.scene_id)            
                    if not os.path.exists(scene_directory):
                            os.mkdir(scene_directory)
                    cam_directory   = "{}/im/scene_{}/{}".format(output_directory,ann.scene_id,cam_name)            
                    if not os.path.exists(cam_directory):
                            os.mkdir(cam_directory)
                   
                    
                   
                    im_path = "{}/{}.png".format(cam_directory,str(f_idx).zfill(4))
                    if not os.path.exists(im_path):
                        frame = ann.buffer[ann.buffer_frame_idx][c_idx].copy() # second item is timestamp or dummy value
                        frame = cv2.resize(frame,(1920,1080))
                        cv2.imwrite(im_path,frame)
                      
                        
                      
                    # generate label path
                    lab_directory = "{}/label".format(output_directory)
                    if not os.path.exists(lab_directory):
                            os.mkdir(lab_directory)
                    scene_directory = "{}/label/scene_{}".format(output_directory,ann.scene_id)            
                    if not os.path.exists(scene_directory):
                            os.mkdir(scene_directory)
                    cam_directory   = "{}/label/scene_{}/{}".format(output_directory,ann.scene_id,cam_name)            
                    if not os.path.exists(cam_directory):
                            os.mkdir(cam_directory)
                            
                    label_directory = "{}/label/scene_{}/{}".format(output_directory,scene_id,cam_name)
                    if not os.path.exists(label_directory):
                        os.mkdir(label_directory)
                    lab_path = "{}/{}.cpkl".format(label_directory,str(f_idx).zfill(4))
                    
                    with open(lab_path,"wb") as f:
                        pickle.dump(cam_frame_annotations,f)
                    
                    # after all objects have been  added
                    all_data.append([im_path,lab_path])
        
       
                    
            
        
            # advance cameras
            ann.next()
            if f_idx % 100 == 0:
                t_est = time.time() - start_time
                print("Cached scene {} frame {}, {} fps cache rate".format(scene_id,f_idx,f_idx/t_est))
    
            # if f_idx > 10:
            #     break
        
    
    # else: # add additional bonus frames
    #     bonus_path = "/home/worklab/Documents/datasets/more_3D_frames"
        
    #     ims = os.listdir(bonus_path + "/im")
    #     labels = os.listdir(bonus_path + "/label_export")
        
    #     for im_path in ims:
    #         label_path = im_path.split(".png")[0] + ".cpkl"
    #         if label_path in labels:
    #             all_data.append([os.path.join(bonus_path, "im", im_path),os.path.join(bonus_path, "label_export", label_path)])

     
    # save_label_path = "{}/data_summary.cpkl".format(output_directory)
    # with open(save_label_path,"wb") as f:
    #     pickle.dump(all_data,f)
    
if __name__ == "__main__":

    video_dir = "/home/worklab/Documents/datasets/I24-3D/video/"
    data_dir  = "/home/worklab/Documents/datasets/I24-3D/data"
    out_dir = "/home/worklab/Documents/datasets/I24-3D/cache"
    
    if False:
        for scene_id in [1,2,3]:
            start_frame = 0
            ann = Scene(video_dir,data_dir,scene_id = scene_id,start_frame=start_frame)
            ann.correct_curve()
            ann.buffer_lim = 10
            #ann = None
            cache_frames(ann,output_directory = out_dir,start_frame = start_frame)
            del ann
            gc.collect()

    directories = [["/home/worklab/Documents/datasets/I24-3D/cache/im/scene_1","/home/worklab/Documents/datasets/I24-3D/cache/label/scene_1"],
                   ["/home/worklab/Documents/datasets/I24-3D/cache/im/scene_2","/home/worklab/Documents/datasets/I24-3D/cache/label/scene_2"],
                   ["/home/worklab/Documents/datasets/I24-3D/cache/im/scene_3","/home/worklab/Documents/datasets/I24-3D/cache/label/scene_3"],
                   ["/home/worklab/Documents/datasets/more_3D_frames/im"      ,"/home/worklab/Documents/datasets/more_3D_frames/label_export"]]
    create_summary(directories,out_dir)