from scene import Scene

import numpy as np
import torch
import _pickle as pickle
import copy
import cv2
import os
import time

start_time = time.time()
all_data = []

def cache_frames(ann,output_directory):
    
    
    # for each frame idx
    for f_idx in range (2700):
        
        if len(ann.data[f_idx]) == 0: # past last frame
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
    
    
        if f_idx > 10:
            break
     
    save_label_path = "{}/data_summary.cpkl".format(output_directory)
    with open(save_label_path,"wb") as f:
        pickle.dump(all_data,f)
    
if __name__ == "__main__":

    video_dir = "/home/worklab/Documents/I24-3D/video"
    data_dir  = "/home/worklab/Documents/I24-3D/data"
    out_dir = "/home/worklab/Documents/I24-3D/cache"
    
    for scene_id in [1,2,3]:
        ann = Scene(video_dir,data_dir,scene_id = scene_id)
        cache_frames(ann,output_directory = out_dir)
