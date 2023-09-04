from ICCV_manual_labeler_ocs import Annotator

import numpy as np
import torch
import _pickle as pickle
import copy
import cv2
import os
import time

extension_distance = 500
start_time = time.time()
all_data = []

# for each dataset
for scene_id in [0,4,6]:
        
    directory = "/home/derek/Data/cv/video/ground_truth_video_06162021/segments_4k"
    directory = "/home/derek/Data/dataset_beta/sequence_{}".format(scene_id)
    #dataset_save_path = "/home/derek/Data/cv/internal_datasets/dataset_beta_image_cache"
    dataset_save_path = "/home/derek/Data/ICCV_2023/im_cache"
    if scene_id == 0:
        exclude_p3c6 = True
    else:
        exclude_p3c6 = False
    
    
    # open the annotator
    ann = Annotator(directory,scene_id = scene_id,exclude_p3c6 = exclude_p3c6)  
    ann.buffer_lim = 1

    with open("ICCV_splines_augmented_{}.cpkl".format(scene_id),"rb") as f:
            [ann.data,ann.all_ts,ann.splines] = pickle.load(f)
    
    # for each frame idx
    for f_idx in range (2700):
        
        if len(ann.data[f_idx]) == 0:
            break
        
        # for each camera
        
        for c_idx in range(len(ann.cameras)):
            cam_frame_annotations = []
            
            camera = ann.cameras[c_idx]
            cam_name = ann.cameras[c_idx].name
            
            # get set of objects labeled within that camera
            ts_data = list(ann.data[ann.frame_idx].values())
            ts_data = list(filter(lambda x: x["camera"] == camera.name,ts_data))
            
            if len(ts_data) == 0:
                break
            
            #ts_data = list(filter(lambda x: x["id"] == self.get_unused_id() - 1,ts_data))

            if True:
                 ts_data = [ann.offset_box_y(copy.deepcopy(obj),reverse = True) for obj in ts_data]
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
            cam_directory = "{}/im/scene_{}/{}".format(dataset_save_path,scene_id,cam_name)
            # if not os.path.exists(cam_directory):
            #     os.mkdir(cam_directory)
            im_path = "{}/{}.png".format(cam_directory,str(f_idx).zfill(4))
            
            
            if True: # I already wrote the images
                if not os.path.exists(im_path):
                    frame = ann.buffer[ann.buffer_frame_idx][c_idx][0].copy() # second item is timestamp
                    cv2.imwrite(im_path,frame)
                
            
            # generate label path
            lab_directory = "{}/label/scene_{}/{}".format(dataset_save_path,scene_id,cam_name)
            if not os.path.exists(lab_directory):
                os.mkdir(lab_directory)
            lab_path = "{}/{}.cpkl".format(lab_directory,str(f_idx).zfill(4))
            
            with open(lab_path,"wb") as f:
                pickle.dump(cam_frame_annotations,f)
            
            # after all objects have been  added
            all_data.append([im_path,lab_path])
            
            # end one iteration of c_idx loop

        # advance cameras
        ann.next()
        if f_idx % 100 == 0:
            t_est = time.time() - start_time
            print("Cached scene {} frame {} labels and frames, {} fps cache rate".format(scene_id,f_idx,f_idx/t_est))
    
        # end one iteration of f_idx loop
        
        # if f_idx > 100:
        #     break

save_label_path = "{}/data_summary.cpkl".format(dataset_save_path)
with open(save_label_path,"wb") as f:
    pickle.dump(all_data,f)