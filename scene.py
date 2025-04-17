"""
To whom it may concern:
    This codebase got away from me frankly. It's the product of me trying to support about
    4 full devs worth of projects and simultaneously come up with sufficient
    novel work to merit graduation, all while becoming increasingly disillusioned with 
    the world of academia and the nature of a PhD altogether. Please bear this in
    mind and don't judge me too harshly as you wade through the sea of poor comments, missing comments,
    blatantly incorrect / outdated comments, vestigial code and objects and attributes, etc.
    - Derek 
"""


from datareader import Data_Reader, Camera_Wrapper, Camera_Wrapper_vpf
from homography import Homography_Wrapper
import _pickle as pickle
import time
import json
from json import JSONEncoder

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import torch
import re
import cv2 as cv
import string
import copy
import csv
import cv2
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


import torch.multiprocessing as mp
ctx = mp.get_context('spawn')

# filter and CNNs


class Scene:
    """
    Scene provides tools for labeling and correcting predicted labels
    for 3D objects tracked through space across multiple cameras. Camera timestamps
    are assumed to be out of phase and carry some error, which is adjustable 
    within this labeling framework. 

    Each camera and set of labels is in essence a discrete rasterization of continuous,vector-like data.
    Thus, several challenges arise in terms of how to manage out-of-phase and 
    out-of-period discretizations. The following guidelines are adhered to:

    i.  We base labels drawn in each camera view on the timestamp of that camera
    ii. We advance the first camera at each "frame", and adjust all other cameras 
        to be within 1/60th second of this time
    iii. We do not project labels from one camera into another
    iv. For most changes, we carry the change forward to all future frames in 
        the same camera view. These include:
            - shift in object x and y position
            - change in timestamp bias for a camera
    v.  We treat class and dimensions as constant for each object. 
        Adjusting these values adjusts them at all times across all cameras
    vi. When interpolating boxes, we assume constant velocity in space (ft)
    vii. We account for time bias once. Since we do not draw boxes across cameras,
         time bias is never used for plotting in this tool, but will be useful
         for labels later down the lien
    """

    def __init__(self, video_dir, data_dir, scene_id=1,start_frame = 0):

        # # get data
        # dr = Data_Reader(data,None,metric = False)
        # self.data = dr.data.copy()
        # del dr

        # # add camera tag to data keys
        # new_data = []
        # for frame_data in self.data:
        #     new_frame_data = {}
        #     for obj in frame_data.values():
        #         key = "{}_{}".format(obj["camera"],obj["id"])
        #         new_frame_data[key] = obj
        #     new_data.append(new_frame_data)
        # self.data = new_data

        self.scene_id = scene_id

        # median class dimesions (in feet of course)
        self.class_dims = {
                "sedan":[16,6,4],
                "midsize":[18,6.5,5],
                "van":[20,6,6.5],
                "pickup":[20,6,5],
                "semi":[55,9,12],
                "truck (other)":[25,9,12],
                "truck": [25,9,12],
                "motorcycle":[7,3,4],
                "trailer":[16,7,3],
                "other":[18,6.5,5]
            }
        
        # class to index and index to class lookup dict
        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    # "motorcycle":6,
                    # "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    # 6:"motorcycle",
                    # 7:"trailer"
                    }

        # get sequences
        sequence_directory = os.path.join(video_dir,"scene{}".format(scene_id))
        self.sequences = {}
        for idx, sequence in enumerate(os.listdir(sequence_directory)):
            if "p3c6" not in sequence or scene_id != 1:
                cap = Camera_Wrapper_vpf(os.path.join(
                    sequence_directory, sequence),ctx,ds=1,gpu = np.random.randint(0,3),start_frame = start_frame)
                # cap = Camera_Wrapper(os.path.join(
                #     sequence_directory, sequence),ds=1)
                self.sequences[cap.name] = cap

        # # get homography
        # hid = "" if homography_id == 1 else "2"
        # with open("EB_homography{}.cpkl".format(hid), "rb") as f:
        #     hg1 = pickle.load(f)
        # with open("WB_homography{}.cpkl".format(hid), "rb") as f:
        #     hg2 = pickle.load(f)
        # TODO - rewrite
        hg_path = os.path.join(data_dir,"hg/scene{}_hg.json".format(scene_id))
        self.hg = Homography_Wrapper(save_file = hg_path)

        # sorted sequence list
        self.seq_keys = list(self.sequences.keys())
        self.seq_keys.sort()

        # # get ts biases
        # try:
        #     self.ts_bias = np.array([list(self.data[0].values())[0]["ts_bias"][key] for key in self.seq_keys])
        # except:
        #     for k_idx,key in enumerate(self.seq_keys):
        #         if key in  list(self.data[0].values())[0]["ts_bias"].keys():
        #             self.ts_bias[k_idx] = list(self.data[0].values())[0]["ts_bias"][key]

        self.cameras = [self.sequences[key] for key in self.seq_keys]
        [next(camera) for camera in self.cameras]
        self.active_cam = 0



        try:
            self.data_path = os.path.join(data_dir,"obj","scene{}_annotations.csv".format(scene_id))
            ts_path = os.path.join(data_dir,"ts","scene{}_ts.csv".format(scene_id))

            self.load_data_csv(self.data_path)
            #self.data = []
            
            self.load_ts_csv(ts_path)
        except:
            print("Did not successfully reload data")
            self.data = []
            self.ts_bias = np.zeros(len(self.seq_keys))
            self.all_ts = []

        try:
            spl_json_path = os.path.join(data_dir,"spl_obj","scene{}_splobj.json".format(scene_id))
            with open(spl_json_path,"r") as f:
                #self.spline_data = json.load(f)
                print("Loaded {} as spline object data".format(spl_json_path))
                self.spline_data = []

        except:
            print("Did not successfully reload spline object data")

        # get length of cameras, and ensure data is long enough to hold all entries
        self.max_frames = max([len(camera) for camera in self.cameras])
        while len(self.data) < self.max_frames:
            self.data.append({})

        # remove all data older than 1/60th second before last camera timestamp
        # max_cam_time = max([cam.ts for cam in self.cameras])
        # if not overwrite:
        #     while list(self.data[0].values())[0]["timestamp"] + 1/60.0 < max_cam_time:
        #         self.data = self.data[1:]

        # get first frames from each camera according to first frame of data
        self.frame_idx = start_frame
        
        self.buffer_frame_idx = -1
        self.buffer_lim = 500
        self.last_frame = 2700
        self.buffer = [[] for _ in range(start_frame)]

       
        self.advance_all()

        self.cont = True
        self.new = None
        self.clicked = False
        self.clicked_camera = None
        self.TEXT = True
        self.LANES = False

        self.active_command = "DIMENSION"
        self.right_click = False
        self.copied_box = None

        self.label_buffer = copy.deepcopy(self.data)

        self.colors = np.random.rand(2000, 3)

        self.stride = 20
        self.plot_idx = 0
        self.PLAY_PAUSE = False

        ranges = {}
        for cam in self.cameras:
            cam = cam.name

            space_pts1 = self.hg.hg1.correspondence[cam]["space_pts"]
            space_pts2 = self.hg.hg2.correspondence[cam]["space_pts"]
            space_pts = np.concatenate((space_pts1, space_pts2), axis=0)

            minx = np.min(space_pts[:, 0])
            maxx = np.max(space_pts[:, 0])

            ranges[cam] = [minx, maxx]
        self.ranges = ranges

        
        self.MASK = False
        self.mask_ims = {}
        mask_dir = os.path.join(data_dir,"mask","scene{}".format(scene_id))
        mask_paths = os.listdir(mask_dir)
        for path in mask_paths:
            if "1080" in path:
                key = path.split("_")[0]
                path = os.path.join(mask_dir, path)
                im = cv2.imread(path)

                self.mask_ims[key] = im
                
    def correct_curve(self):
        for p in [1,3]:
            try:
                self.hg.hg2.correspondence["p{}c4".format(p)]["curve"] = self.hg.hg1.correspondence["p{}c4".format(p)]["curve"]
            except:
                print("Could not correct curve porams for p{}c4".format(p))
        for p in [2]:
            try:
                self.hg.hg2.correspondence["p{}c3".format(p)]["curve"] = self.hg.hg1.correspondence["p{}c3".format(p)]["curve"]
            except:
                print("Could not correct curve porams for p{}c3".format(p))

    def safe(self, x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x

    def save_data_csv(self,csv_path):
            
        """ 
        data is a list where one datum is all the data for a frame as a dictionary. 
        The key of said dictionary is ID_camera. Let's port it into a csv why don't we?
        """
        print("Saving to {}... ".format(csv_path,))

        data = self.data

        header = ["camera","id","x","y","l","w","h","direction","class","gen","cab_length"] 
        
        
        all_rows = []
        for fidx,frame_data in enumerate(data):
                
            for obj in frame_data.values():
                if "cab_length" not in obj.keys():
                    obj["cab_length"] = 0
                    
                row = [fidx] 
                for key in header:
                    try:
                        if key == "gen":
                            row.append(obj[key].lower())
                            
                        else:
                            try:
                                row.append(self.safe(np.round(self.safe(obj[key]),3)))
                            except:
                                row.append(self.safe(obj[key]))
                            
                            
                    except KeyError:
                        if key == "gen":
                            row.append("manual")
                
                all_rows.append(row)
        
        # write rows
        header = ["frame"] + header    
        with open(csv_path,"w") as f:
            
            csvwriter = csv.writer(f) 
            csvwriter.writerow(header)
            csvwriter.writerows(all_rows)
        print("Wrote {} with {} annotations".format(csv_path,len(all_rows)))

    def load_data_csv(self,csv_path):
        lines = []
        with open(csv_path,"r") as f:
            csv_reader = csv.reader(f, delimiter=',')
            
            for line in csv_reader:
                lines.append(line)
        
        headers = lines[0] 
        del lines[0]
        
        data = []
        cur_fidx = 0
        frame_data = {}
        for line in lines:
            datum = {}
            for kidx,key in enumerate(headers):
            
                datum[key] = line[kidx]
                if key in ["x","y","l","w","h","cab_length"]:
                    datum[key] = float(datum[key])
                elif key in ["frame","id","direction"]:
                    datum[key] = int(datum[key])
                
            obj_id = "{}_{}".format(datum["camera"],datum["id"])
            if datum["frame"] == cur_fidx:
                frame_data[obj_id] = datum
            else:
                data.append(frame_data)
                frame_data = {}
                cur_fidx += 1
                frame_data[obj_id] = datum

        self.data = data
        print("Loaded {} as data".format(csv_path))
        
               
    
    def load_ts_csv(self,ts_path):
        lines = []
        with open(ts_path,"r") as f:
            csv_reader = csv.reader(f, delimiter=',')
            
            for line in csv_reader:
                lines.append(line)
        
        headers = lines[0] 
        del lines[0]
        
        data = []
        for lidx,line in enumerate(lines):
            datum = {}
            for kidx,key in enumerate(headers):
                if kidx > 0:
                    datum[key] = float(line[kidx])
                    
                else:
                    assert lidx == int(line[0]) #make sure line index = frame_index recorded
                
                
            data.append(datum)
            datum = {}


        self.all_ts = data
        print("Loaded {} as timestamps".format(ts_path))
        

    def count(self):
        count = 0
        for frame_data in self.data:
            for key in frame_data.keys():
                count += 1
        print("{} total boxes".format(count))

    def toggle_cams(self, dir):
        """dir should be -1 or 1"""

        if self.active_cam + dir < len(self.seq_keys) - 1 and self.active_cam + dir >= 0:
            self.active_cam += dir
            self.plot()


        if self.cameras[self.active_cam].name in ["p1c3", "p1c4", "p2c3", "p2c4", "p3c3", "p3c4"]:
            self.stride = 10
        else:
            self.stride = 20


    def advance_all(self):
        for c_idx, camera in enumerate(self.cameras):
            next(camera)

        frames = [cam.frame for cam in self.cameras]

    

        self.buffer.append(frames)
        if len(self.buffer) > self.buffer_lim:
            #self.buffer = self.buffer[1:]
            del self.buffer[0]

    def fill_buffer(self, n):
        for i in range(n):
            self.next()
            if i % 100 == 0:
                print("On frame {}".format(self.frame_idx))
        self.plot()
        print("Done")

    def next(self):
        """
        Advance a "frame"
        """
        self.label_buffer = None



        if self.frame_idx < len(self.data) and self.frame_idx < self.last_frame:
            self.frame_idx += 1

            # if we are in the buffer, move forward one frame in the buffer
            if self.buffer_frame_idx < -1:
                self.buffer_frame_idx += 1

            # if we are at the end of the buffer, advance frames and store
            else:
                # advance cameras
                self.advance_all()
        else:
            print("On last frame")

    def prev(self):
        self.label_buffer = None



        if self.frame_idx > 0 and self.buffer_frame_idx > -self.buffer_lim:
            self.frame_idx -= 1
            self.buffer_frame_idx -= 1
        else:
            print("Cannot return to previous frame. First frame or buffer limit")

    
    def spl_boxes(self,interp_time):
        """
        Interpolate positions for all active spline objects at the given time
        """
        spl_boxes = []
        
        for item in spl_boxes:
            if item["timestamp"][0] <= interp_time and item["timestamp"][-1] >= interp_time:
                
                idx1 = 0
                # find time that is 
                
                while item["timestamp"][idx1+1] < interp_time and  idx1 < len(item["timestamp"])-2:
                    idx1 += 1
                    
                t1 = item["timestamp"][idx1]
                t2 = item["timestamp"][idx1+1]
                x1 = item["x"][idx1]
                x2 = item["x"][idx1 + 1]
                y1 = item["y"][idx1]
                y2 = item["y"][idx1 + 1]
                
                r1 = (t2 - interp_time)/(t2-t1 + 1e-05)
                r2 = 1 - r1
                
                x = x1*r1 + x2*r2
                y = y1*r1 + y2*r2
                
                obj = item.copy()
                obj["x"] = x
                obj["y"] = y
                obj["timestamp"] = interp_time
    
                spl_boxes.append(obj)
        return spl_boxes
    

    def plot(self, extension_distance=200):
        plot_frames = []
        ranges = self.ranges

        for i in range(self.active_cam, self.active_cam+2):
        #for i in range(len(self.cameras)):
            camera = self.cameras[i]
            #cam_ts_bias = self.ts_bias[i]  # TODO!!!

            frame = self.buffer[self.buffer_frame_idx][i].copy()
            # frame = frame.copy() * 0 + 255
            frame_ts = self.all_ts[self.frame_idx][camera.name]
            
            # get frame objects
            # stack objects as tensor and aggregate other data for label
            ts_data = list(self.data[self.frame_idx].values())
            ts_data = list(
                filter(lambda x: x["camera"] == camera.name, ts_data))

            for item in ts_data:
                if "gen" not in item.keys():
                    item["gen"] = "manual"
            ts_data_spline = list(
                filter(lambda x: x["gen"] == "spline", ts_data))
            ts_data = list(filter(lambda x: x["gen"] != "spline", ts_data))

            cabs = []
            for item in ts_data:
                if "cab_length" in item.keys() and item["cab_length"] > 0:
                    cabs.append(item)
            
            if False:
                ts_data = [self.offset_box_y(copy.deepcopy(
                    obj), reverse=True) for obj in ts_data]
                ts_data_spline = [self.offset_box_y(copy.deepcopy(
                    obj), reverse=True) for obj in ts_data_spline]

            # plot non-spline boxes
            ids = [item["id"] for item in ts_data]
            if len(ts_data) > 0:
                boxes = torch.stack([torch.tensor(
                    [obj["x"], obj["y"], obj["l"], obj["w"], obj["h"], obj["direction"]]).float() for obj in ts_data])

                # convert into image space
                im_boxes = self.hg.state_to_im(boxes, name=camera.name)


                # plot on frame
                frame = self.hg.plot_state_boxes(frame, boxes, name=camera.name, color=(
                    0, 150, 0), secondary_color=(0, 150, 0), thickness=2, jitter_px=0)
                
                if True and len(cabs) > 0:
                    ### Cab boxes
                    boxes2 = torch.stack([torch.tensor([item["x"]+item["l"]*item["direction"] - item["cab_length"]*item["direction"],item["y"],item["cab_length"],item["w"],item["h"],item["direction"]]).float() for item in cabs])
    
    
    
                    # plot on frame
                    frame = self.hg.plot_state_boxes(frame, boxes2, name=camera.name, color=(
                        0, 150, 150), secondary_color=(0, 150, 150), thickness=2, jitter_px=0)


                # plot labels
                if self.TEXT:
                    times = [frame_ts for item in ts_data]
                    classes = [item["class"] for item in ts_data]
                    ids = [item["id"] for item in ts_data]
                    directions = [item["direction"] for item in ts_data]
                    directions = ["WB" if item == -1 else "EB" for item in directions]
                    camera.frame = Data_Reader.plot_labels(
                        None, frame, im_boxes*2, boxes, classes, ids, None, directions, times)
           
            
            # plot spline boxes as a different color
            ids = [item["id"] for item in ts_data_spline]
            if len(ts_data_spline) > 0:
                boxes = torch.stack([torch.tensor([obj["x"], obj["y"], obj["l"], obj["w"],
                                    obj["h"], obj["direction"]]).float() for obj in ts_data_spline])

                # convert into image space
                im_boxes = self.hg.state_to_im(boxes, name=camera.name)
                # plot on frame
                frame = self.hg.plot_state_boxes(frame, boxes, name=camera.name, color=(
                    0, 150, 150), secondary_color=(0, 150, 150), thickness=4, jitter_px=0)

            #     # plot labels for spline boxes
            #     if self.TEXT:
            #         times = [self.all_ts[self.frame_idx][camera.name] for item in ts_data_spline]
            #         classes = [item["class"] for item in ts_data_spline]
            #         ids = [item["id"] for item in ts_data_spline]
            #         directions = [item["direction"] for item in ts_data_spline]
            #         directions = ["WB" if item == -1 else "EB" for item in directions]
            #         camera.frame = Data_Reader.plot_labels(
            #             None, frame, im_boxes, boxes, classes, ids, None, directions, times)

            # lastly, plot interpolated spline boxes themselves
            if True:
                splines = self.spl_boxes(frame_ts)
                splines = list(filter(lambda x: (x["x"] > self.ranges[camera.name][0] and x["x"] < self.ranges[camera.name][1]), ts_data))
                splines = [self.offset_box_y(copy.deepcopy(obj), reverse=True) for obj in splines]

                ids = [item["id"] for item in splines]
                if len(splines) > 0:
                    boxes = torch.stack([torch.tensor([obj["x"], obj["y"], obj["l"], obj["w"], obj["h"], obj["direction"]]).float() for obj in splines])
                    frame = self.hg.plot_state_boxes(frame, boxes, name=camera.name, color=(255,255,255), secondary_color=(255,255,255), thickness=1)            


            # plot mask
            if self.MASK:
                mask_im = self.mask_ims[camera.name]/255
                
                ### TEMP
                mask_im = cv2.resize(mask_im,(3840,2160))
                
                blur_im = cv2.blur(frame, (17, 17))
                frame = frame*mask_im + blur_im * (1-mask_im)*0.7

            frame = cv2.resize(frame,(1920,1080))

            # add frame number and camera
            if True:
                font = cv2.FONT_HERSHEY_SIMPLEX
                header_text = "{} frame {}".format(camera.name, self.frame_idx)
                frame = cv2.putText(frame, header_text,
                                    (30, 30), font, 1, (255, 255, 255), 1)

            # plot lane markings
            if self.LANES:
                for direction in ["_EB", "_WB"]:
                    if direction == "_EB":
                        side_offset = 0
                    else:
                        side_offset =  self.hg.hg2.correspondence[camera.name]["space_pts"][0][1]
                        
                    for lane in [0, 12, 24, 36, 48]:
                        # get polyline coordinates in space
                        
                        if direction == "_EB":
                            
                            p2, p1, p0 = self.hg.hg1.correspondence[camera.name]["curve"]
                        else:
                            p2, p1, p0 = self.hg.hg2.correspondence[camera.name]["curve"]
                            
                        x_curve = np.linspace(-3000, 3000, 6000)
                        y_curve = x_curve*0 + lane + side_offset 
                        z_curve = x_curve * 0
                        curve = np.stack([x_curve, y_curve, z_curve], axis=1)
                        curve = torch.from_numpy(curve).unsqueeze(1)
                        curve_im = self.hg.space_to_im(curve, name=camera.name)

                        mask = ((curve_im[:, :, 0] > 0).int() + (curve_im[:, :, 0] < 1920).int() + (
                            curve_im[:, :, 1] > 0).int() + (curve_im[:, :, 1] < 1080).int()) == 4
                        curve_im = curve_im[mask, :]

                        curve_im = curve_im.data.numpy().astype(int)
                        cv2.polylines(frame, [curve_im],
                                      False, (255, 100, 0), 1)

                    for tick in range(0, 2000, 10):
                        y_curve = np.linspace(
                            0, 48, 4) + side_offset  #+ p0 + p1*tick + p2*tick**2
                        x_curve = y_curve * 0 + tick
                        z_curve = y_curve * 0
                        curve = np.stack([x_curve, y_curve, z_curve], axis=1)
                        curve = torch.from_numpy(curve).unsqueeze(1)
                        curve_im = self.hg.space_to_im(curve, name=camera.name)

                        mask = ((curve_im[:, :, 0] > 0).int() + (curve_im[:, :, 0] < 1920).int() + (
                            curve_im[:, :, 1] > 0).int() + (curve_im[:, :, 1] < 1080).int()) == 4
                        curve_im = curve_im[mask, :]

                        curve_im = curve_im.data.numpy().astype(int)

                        th = 1
                        color = (150,150,150)
                        if tick % 200 == 0:
                            th = 2
                            color = (0,0,255)
                        elif tick % 40 == 0:
                            th = 2

                        cv2.polylines(frame, [curve_im], False, color, th)

            

            #cv2.imwrite("frames/{}.png".format(camera.name),frame)
            plot_frames.append(frame)

        # concatenate frames
        n_ims = len(plot_frames)
        n_row = int(np.round(np.sqrt(n_ims)))
        n_col = int(np.ceil(n_ims/n_row))
        
        rsize = 1080
        csize = 1920
        cat_im = np.zeros([rsize*n_row, csize*n_col, 3]).astype(float)
        for i in range(len(plot_frames)):
            im = plot_frames[i]
            row = i // n_col
            col = i % n_col
            #print(row,col,im.shape,cat_im.shape,n_row,n_col,n_ims)
            cat_im[row*rsize:(row+1)*rsize, col*csize:(col+1)*csize, :] = im

        # view frame and if necessary write to file
        cat_im /= 255.0
        self.plot_frame = cat_im

    def output_vid(self):
        self.LANES = False
        self.TEXT = False
        self.active_cam = len(self.cameras)-2
        self.MASK = True
        
        while self.frame_idx < self.last_frame:
            
            if not os.path.exists("video/{}/{}.png".format(self.scene_id,str(self.frame_idx).zfill(4))):
                self.plot()
    
                max_divisor = max(self.plot_frame.shape[0]/2160,self.plot_frame.shape[1]/3840)
                new_size = int(self.plot_frame.shape[1]/max_divisor),int(self.plot_frame.shape[0]/max_divisor)
                resize_im = cv2.resize(self.plot_frame*255, new_size)
                cv2.imwrite("video/{}/{}.png".format(self.scene_id,
                            str(self.frame_idx).zfill(4)), resize_im)

            self.next()

    def add(self, obj_idx, location):

        xy = self.box_to_state(location)[0, :].data.numpy()

        # create new object
        obj = {
            "x": float(xy[0]),
            "y": float(xy[1]),
            "l": self.hg.hg1.class_dims["midsize"][0],
            "w": self.hg.hg1.class_dims["midsize"][1],
            "h": self.hg.hg1.class_dims["midsize"][2],
            "direction": 1 if xy[1] < 60 else -1,
            "class": "midsize",
            "timestamp": self.all_ts[self.frame_idx][self.clicked_camera],
            "id": obj_idx,
            "camera": self.clicked_camera,
            "gen": "Manual"
        }


        key = "{}_{}".format(self.clicked_camera, obj_idx)
        self.data[self.frame_idx][key] = obj

    def box_to_state(self, point, direction=False):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        point = point.copy()
        # transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
            point[2] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point1 = torch.tensor([point[0], point[1]]).unsqueeze(
            0).unsqueeze(0).repeat(1, 8, 1)
        point2 = torch.tensor([point[2], point[3]]).unsqueeze(
            0).unsqueeze(0).repeat(1, 8, 1)
        point = torch.cat((point1, point2), dim=0)

        state_point = self.hg.im_to_state(
            point, name=cam, heights=torch.tensor([0]))

        return state_point[:, :2]

    def shift(self, obj_idx, box, dx=0, dy=0):

        key = "{}_{}".format(self.clicked_camera, obj_idx)
        item = self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"

        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1, 0] - state_box[0, 0]
            dy = state_box[1, 1] - state_box[0, 1]

        if np.abs(dy) > np.abs(dx):  # shift y if greater magnitude of change
            # shift y for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx, len(self.data)):
                key = "{}_{}".format(self.clicked_camera, obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    item["y"] += dy
                break
        else:
            # shift x for obj_idx in this and all subsequent frames
            for frame in range(self.frame_idx, len(self.data)):
                key = "{}_{}".format(self.clicked_camera, obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    item["x"] += dx
                break

    def change_class(self, obj_idx, cls):
        for camera in self.cameras:
            cam_name = camera.name
            for frame in range(0, len(self.data)):
                key = "{}_{}".format(cam_name, obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    item["class"] = cls
                    

    def dimension(self, obj_idx, box, dx=0, dy=0,cab = False):
        """
        Adjust relevant dimension in all frames based on input box. Relevant dimension
        is selected based on:
            1. if self.right_click, height is adjusted - in this case, a set ratio
               of pixels to height is used because there is inherent uncertainty 
               in pixels to height conversion
            2. otherwise, object is adjusted in the principle direction of displacement vector
        """

        key = "{}_{}".format(self.clicked_camera, obj_idx)
        item = self.data[self.frame_idx].get(key)
        if item is not None:
            item["gen"] = "Manual"

        if dx == 0 and dy == 0:
            state_box = self.box_to_state(box)
            dx = state_box[1, 0] - state_box[0, 0]
            dy = state_box[1, 1] - state_box[0, 1]
            # we say that 50 pixels in y direction = 1 foot of change
            dh = -(box[3] - box[1]) * 0.02
        else:
            dh = dy

        key = "{}_{}".format(self.clicked_camera, obj_idx)

        try:
            l = self.data[self.frame_idx][key]["l"]
            w = self.data[self.frame_idx][key]["w"]
            h = self.data[self.frame_idx][key]["h"]
        except:
            return
        
        try:
            cab_length = self.data[self.frame_idx][key]["cab_length"]
        except:
            cab_length = 0

        if cab:
            relevant_change = max(0,dx + cab_length)
            relevant_key = "cab_length"
            
        elif self.right_click:
            relevant_change = dh + h
            relevant_key = "h"
        elif np.abs(dx) > np.abs(dy):
            relevant_change = dx + l
            relevant_key = "l"
        else:
            relevant_change = dy + w
            relevant_key = "w"

        for camera in self.cameras:
            cam = camera.name
            for frame in range(0, len(self.data)):
                key = "{}_{}".format(cam, obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    item[relevant_key] = relevant_change

        # also adjust the copied box if necessary
        if self.copied_box is not None and self.copied_box[0] == obj_idx:
            self.copied_box[1][relevant_key] = relevant_change

    def copy_paste(self, point):
        if self.copied_box is None:
            obj_idx = self.find_box(point)

            if obj_idx is None:
                return

            state_point = self.box_to_state(point)[0]

            key = "{}_{}".format(self.clicked_camera, obj_idx)
            obj = self.data[self.frame_idx].get(key)

            if obj is None:
                return

            base_box = obj.copy()

            # save the copied box
            self.copied_box = (obj_idx, base_box, [
                               state_point[0], state_point[1]].copy())

        else:  # paste the copied box
            start = time.time()
            state_point = self.box_to_state(point)[0]

            obj_idx = self.copied_box[0]
            new_obj = copy.deepcopy(self.copied_box[1])

            dx = state_point[0] - self.copied_box[2][0]
            dy = state_point[1] - self.copied_box[2][1]
            new_obj["x"] += dx
            new_obj["y"] += dy
            new_obj["x"] = new_obj["x"].item()
            new_obj["y"] = new_obj["y"].item()
            new_obj["timestamp"] = self.all_ts[self.frame_idx][self.clicked_camera]
            new_obj["camera"] = self.clicked_camera
            new_obj["gen"] = "Manual"

            # remove existing box if there is one
            key = "{}_{}".format(self.clicked_camera, obj_idx)
            # obj =  self.data[self.frame_idx].get(key)
            # if obj is not None:
            #     del self.data[self.frame_idx][key]

            self.data[self.frame_idx][key] = new_obj


    def interpolate(self, obj_idx, verbose=True, gen="Interpolation"):

        # self.print_all(obj_idx)

        for cur_cam in self.cameras:
            cam_name = cur_cam.name

            prev_idx = -1
            prev_box = None
            for f_idx in range(0, len(self.data)):
                frame_data = self.data[f_idx]

                # get  obj_idx box for this frame if there is one
                cur_box = None
                for obj in frame_data.values():
                    if obj["id"] == obj_idx and obj["camera"] == cam_name:
                        del cur_box
                        cur_box = copy.deepcopy(obj)
                        break

                if prev_box is not None and cur_box is not None:

                    for inter_idx in range(prev_idx+1, f_idx):

                        # doesn't assume all frames are evenly spaced in time
                        t1 = self.all_ts[prev_idx][cam_name]
                        t2 = self.all_ts[f_idx][cam_name]
                        ti = self.all_ts[inter_idx][cam_name]
                        p1 = float(t2 - ti) / float(t2 - t1)
                        p2 = 1.0 - p1

                        new_obj = {
                            "x": p1 * prev_box["x"] + p2 * cur_box["x"],
                            "y": p1 * prev_box["y"] + p2 * cur_box["y"],
                            "l": prev_box["l"],
                            "w": prev_box["w"],
                            "h": prev_box["h"],
                            "direction": prev_box["direction"],
                            "id": obj_idx,
                            "class": prev_box["class"],
                            "timestamp": self.all_ts[inter_idx][cam_name],
                            "camera": cam_name,
                            "gen": gen
                        }

                        key = "{}_{}".format(cam_name, obj_idx)
                        self.data[inter_idx][key] = new_obj

                # lastly, update prev_frame
                if cur_box is not None:
                    prev_idx = f_idx
                    del prev_box
                    prev_box = copy.deepcopy(cur_box)

        if verbose:
            print("Interpolated boxes for object {}".format(obj_idx))

    def correct_homography_Z(self, box):
        dx = self.safe(box[2]-box[0])
        if dx > 500:
            sign = -1
        else:
            sign = 1
        # get dy in image space
        dy = self.safe(box[3] - box[1])
        delta = 10**(dy/1000.0)

        direction = 1 if self.box_to_state(box)[0, 1] < 60 else -1

        if direction == 1:
            self.hg.hg1.correspondence[self.clicked_camera]["P"][:,
                                                                 2] *= sign*delta
        else:
            self.hg.hg2.correspondence[self.clicked_camera]["P"][:,
                                                                 2] *= sign*delta

    def correct_time_bias(self, box):

        # get relevant camera idx

        if box[0] > 1920:
            camera_idx = self.active_cam + 1
        else:
            camera_idx = self.active_cam

        # get dy in image space
        dy = box[3] - box[1]

        # 10 pixels = 0.001
        self.ts_bias[camera_idx] += dy * 0.0001

        self.plot_all_trajectories()

    def delete(self, obj_idx, n_frames=-1):
        """
        Delete object obj_idx in this and n_frames -1 subsequent frames. If n_frames 
        = -1, deletes obj_idx in all subsequent frames
        """
        frame_idx = self.frame_idx

        stop_idx = frame_idx + n_frames
        if n_frames == -1:
            stop_idx = len(self.data)

        while frame_idx < stop_idx:
            try:
                key = "{}_{}".format(self.clicked_camera, obj_idx)
                obj = self.data[frame_idx].get(key)
                if obj is not None:
                    del self.data[frame_idx][key]
            except KeyError:
                pass
            frame_idx += 1

    def get_unused_id(self):
        all_ids = []
        for frame_data in self.data:
            for item in frame_data.values():
                all_ids.append(item["id"])

        all_ids = list(set(all_ids))

        new_id = 0
        while True:
            if new_id in all_ids:
                new_id += 1
            else:
                return new_id

    def on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
            self.start_point = (x, y)
            self.clicked = True
        elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0], self.start_point[1], x, y])
            self.new = box
            self.clicked = False

            if x > 1920:
                self.clicked_camera = self.seq_keys[self.active_cam+1]
                self.clicked_idx = self.active_cam + 1
            else:
                self.clicked_camera = self.seq_keys[self.active_cam]
                self.clicked_idx = self.active_cam

        # some commands have right-click-specific toggling
        elif event == cv.EVENT_RBUTTONDOWN:
            self.right_click = not self.right_click
            self.copied_box = None

        # elif event == cv.EVENT_MOUSEWHEEL:
        #      print(x,y,flags)

    def find_box(self, point):
        point = point.copy()

        # transform point into state space
        if point[0] > 1920:
            cam = self.seq_keys[self.active_cam+1]
            point[0] -= 1920
        else:
            cam = self.seq_keys[self.active_cam]

        point = torch.tensor([point[0], point[1]]).unsqueeze(
            0).unsqueeze(0).repeat(1, 8, 1)
        state_point = self.hg.im_to_state(
            point, name=cam, heights=torch.tensor([0])).squeeze(0)

        print(point,state_point)
        min_dist = np.inf
        min_id = None

        for box in self.data[self.frame_idx].values():

            dist = (box["x"] - state_point[0])**2 + \
                (box["y"] - state_point[1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = box["id"]

        return min_id

    def keyboard_input(self):
        keys = ""
        letters = string.ascii_lowercase + string.digits + "_"
        while True:
            key = cv2.waitKey(1)
            for letter in letters:
                if key == ord(letter):
                    keys = keys + letter
            if key == ord("\n") or key == ord("\r"):
                break
        return keys

    def quit(self):
        self.cont = False
        cv2.destroyAllWindows()
        for cam in self.cameras:
            cam.release()

        self.save2()

    def undo(self):
        if self.label_buffer is not None:
            self.data[self.frame_idx] = self.label_buffer
            self.label_buffer = None
            self.plot()
        else:
            print("Can't undo")

    def plot_trajectory(self, obj_idx=0):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []

        t0 = min(list(self.all_ts[0].values()))

        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name

            x = []
            y = []
            v = []
            time = []

            for frame in range(0, len(self.data), 10):
                key = "{}_{}".format(cam_name, obj_idx)
                item = self.data[frame].get(key)
                if item is not None:
                    x.append(self.safe(item["x"]))
                    y.append(self.safe(item["y"]))
                    time.append(
                        self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                    length = item["l"]

            if len(time) > 1:
                time = [item - t0 for item in time]

                # finite difference velocity estimation
                v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08)
                     for i in range(1, len(x))]
                v += [v[-1]]

                all_time.append(time)
                all_v.append(v)
                all_x.append(x)
                all_y.append(y)
                all_ids.append(obj_idx)
                all_lengths.append(length)

        fig, axs = plt.subplots(3, sharex=True, figsize=(24, 18))
        colors = self.colors

        for i in range(len(all_v)):

            cidx = all_ids[i]
            mk = ["s", "D", "o"][i % 3]

            axs[0].scatter(all_time[i], all_x[i],
                           c=colors[cidx:cidx+1]/(i % 3+1), marker=mk)
            axs[1].scatter(all_time[i], all_v[i],
                           c=colors[cidx:cidx+1]/(i % 3+1), marker=mk)
            axs[2].scatter(all_time[i], all_y[i],
                           c=colors[cidx:cidx+1]/(i % 3+1), marker=mk)

            axs[0].plot(all_time[i], all_x[i], color=colors[cidx])  # /(i%1+1))
            axs[1].plot(all_time[i], all_v[i], color=colors[cidx])  # /(i%3+1))
            axs[2].plot(all_time[i], all_y[i], color=colors[cidx])  # /(i%3+1))

            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i], all_x[i],
                                all_x2, color=colors[cidx])

            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-150, 150)

            fig.suptitle("Object {}".format(obj_idx))

        plt.show()

    def plot_all_trajectories(self):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []

        t0 = min(list(self.all_ts[0].values()))

        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name

            for obj_idx in range(self.get_unused_id()):
                x = []
                y = []
                v = []
                time = []

                for frame in range(0, len(self.data), 10):
                    key = "{}_{}".format(cam_name, obj_idx)
                    item = self.data[frame].get(key)
                    if item is not None:
                        x.append(self.safe(item["x"]))
                        y.append(self.safe(item["y"]))
                        time.append(
                            self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                        length = item["l"]

                if len(time) > 1:
                    time = [item - t0 for item in time]

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08)
                         for i in range(1, len(x))]
                    v += [v[-1]]

                    all_time.append(time)
                    all_v.append(v)
                    all_x.append(x)
                    all_y.append(y)
                    all_ids.append(obj_idx)
                    all_lengths.append(length)

        fig, axs = plt.subplots(3, sharex=True, figsize=(24, 18))
        colors = self.colors

        for i in range(len(all_v)):

            cidx = all_ids[i]
            mk = ["s", "D", "o"][i % 3]

            # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)

            axs[0].plot(all_time[i], all_x[i], color=colors[cidx])  # /(i%1+1))
            axs[1].plot(all_time[i], all_v[i], color=colors[cidx])  # /(i%3+1))
            axs[2].plot(all_time[i], all_y[i], color=colors[cidx])  # /(i%3+1))

            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i], all_x[i],
                                all_x2, color=colors[cidx])

            axs[2].set(xlabel='time(s)', ylabel='Y-pos (ft)')
            axs[1].set(ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-150, 150)

        plt.show()

    def plot_one_lane(self, lane=(70, 85)):
        all_x = []
        all_y = []
        all_v = []
        all_time = []
        all_ids = []
        all_lengths = []

        t0 = min(list(self.all_ts[0].values()))

        for cam_idx, camera in enumerate(self.cameras):
            cam_name = camera.name

            for obj_idx in range(self.get_unused_id()):
                x = []
                y = []
                v = []
                time = []

                for frame in range(0, len(self.data)):
                    key = "{}_{}".format(cam_name, obj_idx)
                    item = self.data[frame].get(key)
                    if item is not None:

                        y_test = self.safe(item["y"])
                        if y_test > lane[0] and y_test < lane[1]:
                            x.append(self.safe(item["x"]))
                            y.append(self.safe(item["y"]))
                            time.append(
                                self.safe(item["timestamp"]) + self.ts_bias[cam_idx])
                            length = item["l"]

                if len(time) > 1:
                    time = [item - t0 for item in time]

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08)
                         for i in range(1, len(x))]
                    v += [v[-1]]

                    all_time.append(time)
                    all_v.append(v)
                    all_x.append(x)
                    all_y.append(y)
                    all_ids.append(obj_idx)
                    all_lengths.append(length)

        fig, axs = plt.subplots(2, sharex=True, figsize=(24, 18))
        colors = self.colors

        for i in range(len(all_v)):

            cidx = all_ids[i]
            mk = ["s", "D", "o"][i % 3]

            # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
            # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)

            axs[0].plot(all_time[i], all_x[i], color=colors[cidx])  # /(i%1+1))
            try:
                v = np.convolve(v, np.hamming(15), mode="same")
                axs[1].plot(all_time[i], all_v[i],
                            color=colors[cidx])  # /(i%3+1))

            except:
                try:
                    v = np.convolve(v, np.hamming(5), mode="same")
                    axs[1].plot(all_time[i], all_v[i],
                                color=colors[cidx])  # /(i%3+1))
                except:
                    axs[1].plot(all_time[i], all_v[i],
                                color=colors[cidx])  # /(i%3+1))

            all_x2 = [all_lengths[i] + item for item in all_x[i]]
            axs[0].fill_between(all_time[i], all_x[i],
                                all_x2, color=colors[cidx])

            axs[1].set(xlabel='time(s)', ylabel='Velocity (ft/s)')
            axs[0].set(ylabel='X-pos (ft)')

            axs[1].set_ylim(-60, 0)

        plt.show()





   



    def offset_box_y(self, box, reverse=False):
        return box
    
        camera = box["camera"]
        direction = box["direction"]

        x = box["x"]

        direct = "_EB" if direction == 1 else"_WB"
        key = camera + direct

        if direct == "_EB":
            p2, p1, p0 = self.hg.hg1.correspondence[camera]["curve"]
        else:
            p2, p1, p0 = self.hg.hg2.correspondence[camera]["curve"]

        y_offset = x**2*p2 + x*p1 + p0

        # if on the WB side, we need to account for the non-zero location of the leftmost line so we don't shift all the way back to near 0
        if direction == -1:
            y_straight_offset = self.hg.hg2.correspondence[camera]["space_pts"][0][1]
            y_offset -= y_straight_offset

        if not reverse:
            box["y"] -= y_offset
        else:
            box["y"] += y_offset

        return box


    def estimate_ts_bias(self):
        """
        Moving sequentially through the cameras, estimate ts_bias of camera n
        relative to camera 0 (tsb_n = tsb relative to n-1 + tsb_n-1)
        - Find all objects that are seen in both camera n and n-1, and that 
        overlap in x-space
        - Sample p evenly-spaced x points from the overlap
        - For each point, compute the time for each camera tracklet for that object
        - Store the difference as ts_bias estimate
        - Average all ts_bias estimates to get ts_bias
        - For analysis, print statistics on the error estiamtes
        """

        self.ts_bias[0] = 0

        for cam_idx in range(1, len(self.cameras)):
            cam = self.cameras[cam_idx].name
            decrement = 1
            while True:
                prev_cam = self.cameras[cam_idx-decrement].name

                diffs = []

                for obj_idx in range(self.get_unused_id()):

                    # check whether object exists in both cameras and overlaps
                    c1x = []
                    c1t = []
                    c0x = []
                    c0t = []

                    for frame_data in self.data:
                        key = "{}_{}".format(cam, obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c1x.append(self.safe(obj["x"]))
                            c1t.append(self.safe(obj["timestamp"]))

                        key = "{}_{}".format(prev_cam, obj_idx)
                        if frame_data.get(key):
                            obj = frame_data.get(key)
                            c0x.append(self.safe(obj["x"]))
                            c0t.append(self.safe(obj["timestamp"]))

                    if len(c0x) > 1 and len(c1x) > 1 and max(c0x) > min(c1x):

                        # camera objects overlap from minx to maxx
                        minx = max(min(c1x), min(c0x))
                        maxx = min(max(c1x), max(c0x))

                        # get p evenly spaced x points
                        p = 5
                        ran = maxx - minx
                        sample_points = []
                        for i in range(p):
                            point = minx + ran/(p-1)*i
                            sample_points.append(point)

                        for point in sample_points:
                            time = None
                            prev_time = None
                            # estimate time at which cam object was at point
                            for i in range(1, len(c1x)):
                                if (c1x[i] - point) * (c1x[i-1] - point) <= 0:
                                    ratio = (point-c1x[i-1]) / \
                                        (c1x[i]-c1x[i-1] + 1e-08)
                                    time = c1t[i-1] + (c1t[i] - c1t[i-1])*ratio

                            # estimate time at which prev_cam object was at point
                            for j in range(1, len(c0x)):
                                if (c0x[j] - point) * (c0x[j-1] - point) <= 0:
                                    ratio = (point-c0x[j-1]) / \
                                        (c0x[j]-c0x[j-1] + 1e-08)
                                    prev_time = c0t[j-1] + \
                                        (c0t[j] - c0t[j-1])*ratio

                            # relative to previous camera, cam time is diff later when object is at same location
                            if time and prev_time:
                                diff = self.safe(time - prev_time)
                                #diff = np.sign(diff) * np.power(diff,2)
                                diffs.append(diff)

                # after all objects have been considered
                if len(diffs) > 0:
                    diffs = np.array(diffs)
                    #avg_diff = np.sqrt(np.abs(np.mean(diffs))) * np.sign(np.mean(diffs))
                    avg_diff = np.mean(diffs)
                    stdev = np.std(diffs)

                    # since diff is positive if camera clock is ahead, we subtract it such that adding ts_bias to camera timestamps corrects the error
                    abs_bias = self.ts_bias[cam_idx-decrement] - avg_diff

                    print("Camera {} ofset relative to camera {}: {}s ({}s absolute)".format(
                        cam, prev_cam, avg_diff, abs_bias))
                    self.ts_bias[cam_idx] = abs_bias

                    break

                else:

                    print("No matching points for cameras {} and {}".format(
                        cam, prev_cam))
                    decrement += 1
                    if cam_idx - decrement >= 0:
                        prev_cam = self.cameras[cam_idx-2].name
                    else:
                        break
        print("Done)")

    




    def run(self):
        """
        Main processing loop - you can add additional functions in this loop
        """

        cv2.namedWindow("window")
        cv.setMouseCallback("window", self.on_mouse, 0)
        self.plot()

        while(self.cont):  # one frame

            # handle click actions

            if self.new is not None:
                # buffer one change
                self.label_buffer = copy.deepcopy(self.data[self.frame_idx])

                # Add and delete objects
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new)
                    try:
                        n_frames = int(self.keyboard_input())
                    except:
                        n_frames = -1
                    self.delete(obj_idx, n_frames=n_frames)

                elif self.active_command == "ADD":
                    # get obj_idx
                    obj_idx = self.get_unused_id()
                    self.add(obj_idx, self.new)

                # Shift object
                elif self.active_command == "SHIFT":
                    obj_idx = self.find_box(self.new)
                    self.shift(obj_idx, self.new)

                # Adjust object dimensions
                elif self.active_command == "DIMENSION":
                    obj_idx = self.find_box(self.new)
                    self.dimension(obj_idx, self.new)

                # copy and paste a box across frames
                elif self.active_command == "COPY PASTE":
                    self.copy_paste(self.new)

                # interpolate between copy-pasted frames
                elif self.active_command == "INTERPOLATE":
                    obj_idx = self.find_box(self.new)
                    self.interpolate(obj_idx)

                # correct vehicle class
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx, cls)

                # adjust homography
                elif self.active_command == "HOMOGRAPHY":
                    self.correct_homography_Z(self.new)

                elif self.active_command == "2D PASTE":
                    self.paste_in_2D_bbox(self.new)



                self.plot()
                self.new = None

            # Show frame
            cv2.imshow("window", self.plot_frame)
            title = "{} {}     Frame {}/{}, Cameras {} and {}".format("R" if self.right_click else "", self.active_command,
                                                                      self.frame_idx, self.max_frames, self.seq_keys[self.active_cam], self.seq_keys[self.active_cam + 1])
            cv2.setWindowTitle("window", str(title))



            # Handle keystrokes
            key = cv2.waitKey(1)


            if key == ord(" "):
                self.PLAY_PAUSE = not self.PLAY_PAUSE
            if self.PLAY_PAUSE:
                self.next()
                self.plot()
                continue
            
            
            if key == ord('9'):
                self.next()
                self.plot()
            elif key == ord('8'):
                self.prev()
                self.plot()
                
            elif key == ord("q"):
                self.quit()
            elif key == ord("w"):
                self.save_data_csv(self.data_path)
           

            elif key == ord("["):
                self.toggle_cams(-1)
            elif key == ord("]"):
                self.toggle_cams(1)

            elif key == ord("u"):
                self.undo()
            elif key == ord("-"):
                [self.prev() for i in range(self.stride)]
                self.plot()
            elif key == ord("="):
                [self.next() for i in range(self.stride)]
                self.plot()
            elif key == ord("+"):
                print("Filling buffer. Type number of frames to buffer...")
                n = int(self.keyboard_input())
                self.fill_buffer(n)

            elif key == ord("?"):
                self.estimate_ts_bias()
                self.plot_all_trajectories()
            elif key == ord("t"):
                self.TEXT = not self.TEXT
                self.plot()
            elif key == ord("l"):
                self.LANES = not self.LANES
                self.plot()
            elif key == ord("m"):
                self.MASK = not self.MASK
                self.plot()

            elif key == ord("p"):

                try:
                    n = int(self.keyboard_input())
                except:
                    n = self.plot_idx
                self.plot_trajectory(obj_idx=n)
                self.plot_idx = n + 1

            # toggle commands
            elif key == ord("a"):
                self.active_command = "ADD"
            elif key == ord("r"):
                self.active_command = "DELETE"
            elif key == ord("s"):
                self.active_command = "SHIFT"
            elif key == ord("d"):
                self.active_command = "DIMENSION"
            elif key == ord("c"):
                self.active_command = "COPY PASTE"
            elif key == ord("i"):
                self.active_command = "INTERPOLATE"
            elif key == ord("v"):
                self.active_command = "VEHICLE CLASS"
            elif key == ord("h"):
                self.active_command = "HOMOGRAPHY"


            elif self.active_command == "COPY PASTE" and self.copied_box:
                nudge = 0.25
                if key == ord("1") or int(key) == 177:
                    self.shift(self.copied_box[0], None, dx=-nudge)
                    self.plot()
                if key == ord("5") or int(key) == 181:
                    self.shift(self.copied_box[0], None, dy=nudge)
                    self.plot()
                if key == ord("3") or int(key) == 179:
                    self.shift(self.copied_box[0], None, dx=nudge)
                    self.plot()
                if key == ord("2") or int(key) == 178:
                    self.shift(self.copied_box[0], None, dy=-nudge)
                    self.plot()

            elif self.active_command == "DIMENSION" and self.copied_box:
                nudge = 0.1
                if key == ord("1") or key == 177: # 149 is the numpad '1' for me, you may need to change these
                    self.dimension(self.copied_box[0], None, dx=-nudge*2)
                    self.plot()
                if key == ord("5") or key == 181:
                    self.dimension(self.copied_box[0], None, dy=nudge)
                    self.plot()
                if key == ord("3") or key == 179:
                    self.dimension(self.copied_box[0], None, dx=nudge*2)
                    self.plot()
                if key == ord("2") or key == 178:
                    self.dimension(self.copied_box[0], None, dy=-nudge)
                    self.plot()
                if key == ord("0"):
                    nudge = 1
                    self.dimension(self.copied_box[0],None,dx = -nudge,cab = True)
                    self.plot()
                if key == ord("."):
                    nudge = 1
                    self.dimension(self.copied_box[0],None,dx = nudge,cab = True)
                    self.plot()

#%% Here's a bunch of garbage you probably don't need but I left just in case it 
# inspires some good ideas for you


def plot_proj_error(all_errors, bins=100, cutoff_error=20, names=[]):
    plt.figure(figsize=(7, 5))

    colors = np.array([[0, 0, 1], [1, 0, 0], [0, 0.5, 0.7], [
                      0.7, 0.5, 0], [0.5, 0.5, 0], [0.5, 0.5, 0]])
    xmeans = []
    ymeans = []
    count_max = []
    for i, errors in enumerate(all_errors):
        minv = 0
        maxv = max(errors)
        cutoff = cutoff_error

        ran = np.linspace(minv, min(cutoff, maxv), 100)
        count = np.zeros(ran.shape)

        for item in errors:
            binned = False
            for r in range(1, len(ran)):
                if item > ran[r-1] and item < ran[r]:
                    count[r-1] += 1
                    binned = True
                    break
            if not binned:
                count[-1] += 1
        count /= len(errors)

        # remove all 0 bins
        ran = ran[np.where(count > 0)]
        count = count[np.where(count > 0)]

        plt.plot(ran, count, c=colors[i])

        mean = sum(errors)/len(errors)
        # find closest bin
        idx = 0
        while mean > ran[idx]:
            idx += 1
            print(mean, idx, ran[idx], count[idx])

        ymeans.append(count[idx])
        xmeans.append(mean)
        m = np.max(count)
        count_max.append(m)

    for i in range(len(xmeans)-1):
        plt.annotate("{:2f} ft".format(
            xmeans[i]), (xmeans[i], ymeans[i]), rotation=45, fontsize=16)
        plt.axvline(x=xmeans[i], ymax=ymeans[i]/max(count_max),
                    ls=":", c=colors[i], label='_nolegend_')

    plt.xlim([0, cutoff])

    plt.ylim([0, np.max(count)])
    plt.yticks([])

    plt.xticks(fontsize=18)
    plt.legend(names, fontsize=18)
    plt.ylabel("Relative frequency", fontsize=24)
    plt.xlabel("Cross-camera projection error (ft)", fontsize=24)
    plt.savefig("histogram.pdf", bbox_inches="tight")
    plt.show()


def plot_histograms(cutoff_error=[3, 2], n_bins=30):

    # for pixel moving plot
    rescale = 7
    colors = np.array([[1, 0, 0], [0.5, 0, 0.5], [1, 0, 1], [0, 0, 1], [
                      0.3, 0.2, 0.9], [0, 0.5, 0.5], [0, 0.7, 0.2], [.3, 1, 0], [0, 1, 0]])
    directory = "results"
    includex = [1, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    includey = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    includep = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    llx = [20, 20, 35, 45, 110, 105, 0, 0, 0]
    lly = [20, 20, 50, 115, 105, 75, 0, 0, 0]
    llp = [20, 20, 50, 115, 105, 75, 0, 0, 0]

    rescale = 5
    rot = 45
    # colors = np.array([[0.2,0.5,0.2],[1,0,0],[0.5,0,0.5],[0,0,1],[0,0.7,0.7],[0.2,0.6,0.2],[0,0,0],[0,0,0],[0,0,0]])
    # directory = "histogram_data"
    # includex = [1,1,0,1,0,0]
    # includey = [1,1,1,0,0,0]
    # llx      = [20,20,35,45,110,105,0,0,0]
    # lly      = [20,20,50,115,105,75,0,0,0]

    xmeans1 = []
    ymeans1 = []
    count_max1 = []
    xmeans2 = []
    ymeans2 = []
    count_max2 = []

    legend1 = []
    legend2 = []
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    paths = os.listdir(directory)
    paths.sort()

    for f_idx, file in enumerate(paths):
        # get data
        path = os.path.join(directory, file)

        name = path.split("/")[-1].split(".")[0]
        name = name.replace("_", " ")

        try:
            with open(path, "rb") as f:
                [x_err, y_err, p_err, _, _, _] = pickle.load(f)
        except:
            continue

        # plot x data
        clipped = 0
        if includex[f_idx]:

            # plot histogram
            minv = 0
            maxv = max(x_err)
            cutoff = cutoff_error[0]

            ran = np.linspace(minv, cutoff+0.1, n_bins)
            count = np.zeros(ran.shape)

            for item in x_err:
                binned = False
                for r in range(1, len(ran)):
                    if item > ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(x_err)

            # remove all 0 bins
            #ran = ran[np.where(count > 0)]
            #count = count[np.where(count > 0)]

            axs[0].plot(ran, count*rescale, c=colors[f_idx])

            mean = sum(x_err)/len(x_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                # print(mean,idx,ran[idx],count[idx])

            ratio = (mean-ran[idx-1])/(ran[idx] - ran[idx-1])
            ymeans1.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans1.append(mean)
            m = np.max(count)
            count_max1.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend1.append("{} ({:.1f}%)".format(name, clip_percent))

        else:
            ymeans1.append(0)
            xmeans1.append(0)
            count_max1.append(0)

        # plot y data
        clipped = 0
        if includey[f_idx]:

            # plot histogram
            minv = 0
            maxv = max(y_err)
            cutoff = cutoff_error[1]

            ran = np.linspace(minv, cutoff+.1, n_bins)
            count = np.zeros(ran.shape)

            for item in y_err:
                binned = False
                for r in range(1, len(ran)):
                    if item > ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(y_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]

            axs[1].plot(ran, count*rescale, c=colors[f_idx])

            mean = sum(y_err)/len(y_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                # print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] - ran[idx-1])
            ymeans2.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans2.append(mean)
            m = np.max(count)
            count_max2.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend2.append("{} ({:.1f}%)".format(name, clip_percent))
        else:
            ymeans2.append(0)
            xmeans2.append(0)
            count_max2.append(0)

    # plot x means
    for i in range(len(paths)):
        if includex[i]:
            axs[0].annotate("{:.3f} ft".format(xmeans1[i]),
                            xycoords='data',
                            xy=(xmeans1[i], min(ymeans1[i]*rescale, 0.5)),
                            textcoords='offset points',
                            xytext=(xmeans1[i]+llx[i]*np.cos(np.pi/180*rot),
                                    ymeans1[i]+llx[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation=rot,
                            fontsize=14)
            axs[0].axvline(x=xmeans1[i], ymax=ymeans1[i]/max(count_max1)
                           * rescale, ls=":", c=colors[i], label='_nolegend_')

    # plot y means
    for i in range(len(paths)):
        if includey[i]:
            plus = np.random.rand()*60
            axs[1].annotate("{:.3f} ft".format(xmeans2[i]),
                            xycoords='data',
                            xy=(xmeans2[i], min(ymeans2[i]*rescale, 0.5)),
                            textcoords='offset points',
                            xytext=(xmeans2[i]+lly[i]*np.cos(np.pi/180*rot),
                                    ymeans2[i]+lly[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation=rot,
                            fontsize=14)
            axs[1].axvline(x=xmeans2[i], ymax=ymeans2[i]/max(count_max2)
                           * rescale, ls=":", c=colors[i], label='_nolegend_')

    axs[0].set_xlim([0, cutoff_error[0]])
    axs[0].set_ylim([0, np.max(count_max1)])
    axs[1].set_xlim([0, cutoff_error[1]])
    axs[1].set_ylim([0, np.max(count_max2)])

    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)

    axs[0].set_ylabel("Relative frequency", fontsize=18)
    axs[0].set_xlabel("Cross-camera x-error (ft)", fontsize=18)
    axs[1].set_xlabel("Cross-camera y-error (ft)", fontsize=18)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    axs[0].legend(legend1, fontsize=14)
    axs[1].legend(legend2, fontsize=14)

    plt.savefig("histogram.pdf", bbox_inches="tight")
    plt.show()


def plot_histograms2(cutoff_error=[3, 1.5, 25], n_bins=30):

    # for pixel moving plot
    colors = np.array([[1, 0, 0], [0.5, 0, 0.5], [1, 0, 1], [0, 0, 1], [
                      0.3, 0.2, 0.9], [0, 0.5, 0.5], [0.6, 0.7, 0.4], [.3, 1, 0], [0, 1, 0]])
    directory = "results"
    includex = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    includey = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    includep = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    llx = [20, 10, 20, 40, 50, 80, 50, 140, 20]
    lly = [20, 20, 20, 20, 35, 55, 160, 60, 20]
    llp = [20, 40, 10, 30, 40, 130, 80, 160, 50]

    rescale = 5
    rot = 70
    # colors = np.array([[0.2,0.5,0.2],[1,0,0],[0.5,0,0.5],[0,0,1],[0,0.7,0.7],[0.2,0.6,0.2],[0,0,0],[0,0,0],[0,0,0]])
    # directory = "histogram_data"
    # includex = [1,1,0,1,0,0]
    # includey = [1,1,1,0,0,0]
    # llx      = [20,20,35,45,110,105,0,0,0]
    # lly      = [20,20,50,115,105,75,0,0,0]

    xmeans1 = []
    ymeans1 = []
    count_max1 = []
    xmeans2 = []
    ymeans2 = []
    count_max2 = []
    xmeans3 = []
    ymeans3 = []
    count_max3 = []

    legend1 = []
    legend2 = []
    legend3 = []
    fig, axs = plt.subplots(1, 3, figsize=(21, 5))
    paths = os.listdir(directory)
    paths.sort()

    for f_idx, file in enumerate(paths):
        # get data
        path = os.path.join(directory, file)

        name = path.split("/")[-1].split(".")[0][4:]
        name = name.replace("_", " ")

        try:
            with open(path, "rb") as f:
                [x_err, y_err, p_err, _, _, _] = pickle.load(f)
        except:
            with open(path, "rb") as f:
                [x_err, y_err, p_err, _, _] = pickle.load(f)

        # plot x data
        clipped = 0
        if includex[f_idx]:

            # plot histogram
            minv = 0
            maxv = max(x_err)
            cutoff = cutoff_error[0]

            ran = np.linspace(minv, cutoff+0.1, n_bins)
            count = np.zeros(ran.shape)

            for item in x_err:
                binned = False
                for r in range(1, len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned and item > ran[-1]:
                    clipped += 1
            count /= len(x_err)

            # remove all 0 bins
            #ran = ran[np.where(count > 0)]
            #count = count[np.where(count > 0)]

            axs[0].plot(ran, count*rescale, c=colors[f_idx])

            mean = sum(x_err)/len(x_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                # print(mean,idx,ran[idx],count[idx])

            ratio = (mean-ran[idx-1])/(ran[idx] - ran[idx-1])
            ymeans1.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans1.append(mean)
            m = np.max(count)
            count_max1.append(m)
            clip_percent = int((1-clipped/len(x_err)) * 1000)/10
            legend1.append("{} ({:.1f}%)".format(name, clip_percent))

        else:
            ymeans1.append(0)
            xmeans1.append(0)
            count_max1.append(0)

        # plot y data
        clipped = 0
        if includey[f_idx]:

            # plot histogram
            minv = 0
            maxv = max(y_err)
            cutoff = cutoff_error[1]

            ran = np.linspace(minv, cutoff+.1, n_bins)
            count = np.zeros(ran.shape)

            for item in y_err:
                binned = False
                for r in range(1, len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(y_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]

            axs[1].plot(ran, count*rescale, c=colors[f_idx])

            mean = sum(y_err)/len(y_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                # print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] - ran[idx-1])
            ymeans2.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans2.append(mean)
            m = np.max(count)
            count_max2.append(m)
            clip_percent = int((1-clipped/len(y_err)) * 1000)/10
            legend2.append("{} ({:.1f}%)".format(name, clip_percent))
        else:
            ymeans2.append(0)
            xmeans2.append(0)
            count_max2.append(0)

        # plot pixel data
        clipped = 0
        if includep[f_idx]:

            # plot histogram
            minv = 0
            maxv = max(p_err)
            cutoff = cutoff_error[2]

            ran = np.linspace(minv, cutoff+1, n_bins)
            count = np.zeros(ran.shape)

            for item in p_err:
                binned = False
                for r in range(1, len(ran)):
                    if item >= ran[r-1] and item < ran[r]:
                        count[r-1] += 1
                        binned = True
                        break
                if not binned:
                    clipped += 1
            count /= len(p_err)
            # remove all 0 bins
            # ran = ran[np.where(count > 0)]
            # count = count[np.where(count > 0)]

            axs[2].plot(ran, count*rescale, c=colors[f_idx])

            mean = sum(p_err)/len(p_err)
            # find closest bin
            idx = 0
            while mean > ran[idx]:
                idx += 1
                # print(mean,idx,ran[idx],count[idx])
            ratio = (mean-ran[idx-1])/(ran[idx] - ran[idx-1])
            ymeans3.append((count[idx]*ratio+count[idx-1]*(1-ratio)))
            xmeans3.append(mean)
            m = np.max(count)
            count_max3.append(m)
            clip_percent = int((1-clipped/len(p_err)) * 1000)/10
            legend3.append("{} ({:.1f}%)".format(name, clip_percent))
        else:
            ymeans3.append(0)
            xmeans3.append(0)
            count_max3.append(0)

    # plot x means
    for i in range(len(paths)):
        if includex[i]:
            axs[0].annotate("{:.2f} ft".format(xmeans1[i]),
                            xycoords='data',
                            xy=(xmeans1[i], min(ymeans1[i]*rescale, 0.5)),
                            textcoords='offset points',
                            xytext=(xmeans1[i]+llx[i]*np.cos(np.pi/180*rot),
                                    ymeans1[i]+llx[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation=rot,
                            fontsize=14)
            axs[0].axvline(x=xmeans1[i], ymax=ymeans1[i]/max(count_max1)
                           * rescale, ls=":", c=colors[i], label='_nolegend_')

    # plot y means
    for i in range(len(paths)):
        if includey[i]:
            axs[1].annotate("{:.2f} ft".format(xmeans2[i]),
                            xycoords='data',
                            xy=(xmeans2[i], min(ymeans2[i]*rescale, 0.5)),
                            textcoords='offset points',
                            xytext=(xmeans2[i]+lly[i]*np.cos(np.pi/180*rot),
                                    ymeans2[i]+lly[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation=rot,
                            fontsize=14)
            axs[1].axvline(x=xmeans2[i], ymax=ymeans2[i]/max(count_max2)
                           * rescale, ls=":", c=colors[i], label='_nolegend_')

    # plot p means
    for i in range(len(paths)):
        rot = 65
        if includep[i]:
            axs[2].annotate("{:.1f} px".format(xmeans3[i]),
                            xycoords='data',
                            xy=(xmeans3[i], min(ymeans3[i]*rescale, 0.5)),
                            textcoords='offset points',
                            xytext=(xmeans3[i]+llp[i]*np.cos(np.pi/180*rot),
                                    ymeans3[i]+llp[i]*np.sin(np.pi/180*rot)),
                            arrowprops=dict(arrowstyle='-', color='black'),
                            rotation=rot,
                            fontsize=14)
            axs[2].axvline(x=xmeans3[i], ymax=ymeans3[i]/max(count_max3)
                           * rescale, ls=":", c=colors[i], label='_nolegend_')

    axs[0].set_xlim([0, cutoff_error[0]])
    axs[0].set_ylim([0, np.max(count_max1)])
    axs[1].set_xlim([0, cutoff_error[1]])
    axs[1].set_ylim([0, np.max(count_max2)])
    axs[2].set_xlim([0, cutoff_error[2]])
    axs[2].set_ylim([0, np.max(count_max3)])

    axs[0].set_yticks([])
    axs[1].set_yticks([])
    axs[2].set_yticks([])
    axs[0].xaxis.set_tick_params(labelsize=14)
    axs[1].xaxis.set_tick_params(labelsize=14)
    axs[2].xaxis.set_tick_params(labelsize=14)

    axs[0].set_ylabel("Relative frequency", fontsize=18)
    axs[0].set_xlabel("a.) Cross-camera x-error (ft)", fontsize=24)
    axs[1].set_xlabel("b.) Cross-camera y-error (ft)", fontsize=24)
    axs[2].set_xlabel("c.) Cross-camera pixel error", fontsize=24)

    plt.subplots_adjust(wspace=0.1, hspace=0)

    axs[0].legend(legend1, fontsize=14)
    axs[1].legend(legend2, fontsize=14)
    axs[2].legend(legend3, fontsize=14)

    plt.savefig("histogram.pdf", bbox_inches="tight")
    plt.show()






def plot_trajectories_unified(anns, lane=[82, 98],
                              selected_idx=4,
                              smooth=1,
                              v_range=[0, 140],
                              a_range=[-20, 20],
                              theta_range=[-25, 25]):

    fig, axs = plt.subplots(2, 4, figsize=(25, 5*len(anns)), sharex=True)
    t0 = min(list(anns[0].all_ts[0].values()))
    for a_idx, ann in enumerate(anns):
        ax1, ax2, ax3, ax4 = axs[a_idx, 0:4]

        all_x = []
        all_y = []
        all_time = []
        all_ids = []
        all_lengths = []

        all_v = []
        all_a = []
        all_theta = []

        for obj_idx in range(ann.get_unused_id()):
            x = []
            y = []
            time = []

            for cam_idx, camera in enumerate(ann.cameras):
                cam_name = camera.name
                for frame in range(0, len(ann.data)):
                    key = "{}_{}".format(cam_name, obj_idx)
                    item = ann.data[frame].get(key)
                    if item is not None:
                        y_test = ann.safe(item["y"])
                        if y_test > lane[0] and y_test < lane[1]:
                            x.append(ann.safe(item["x"]))
                            y.append(ann.safe(item["y"]))
                            time.append(ann.safe(item["timestamp"]))
                            length = item["l"]

            if len(time) == 0:
                continue

            # sort by time
            x = np.array(x)
            y = np.array(y)
            time = np.array(time)  # - t0

            order = np.argsort(time)

            x = x[order]
            y = y[order]
            time = time[order]

            keep = [0]
            for i in range(1, len(time)):
                if time[i] >= time[i-1] + 0.01:
                    keep.append(i)

            x = x[keep]
            y = y[keep]
            time = time[keep]

            # estimate derivative qualities
            if len(time) > 1:

                try:
                    vel_spline = ann.splines[obj_idx][0].derivative()
                    v = vel_spline(time)

                except:

                    # finite difference velocity estimation
                    v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08)
                         for i in range(1, len(x))]
                    v += [v[-1]]
                    v = np.array(v)
                    #v = np.convolve(v,np.hamming(smooth),mode = "same")
                v *= -1

                vy = [(y[i] - y[i-1]) / (time[i] - time[i-1] + 1e-08)
                      for i in range(1, len(y))]
                vy += [vy[-1]]
                vy = np.array(vy)
                vy = np.convolve(vy, np.hamming(smooth), mode="same")

                try:
                    a_spline = ann.splines[obj_idx][0].derivative(
                    ).derivative()
                    a = a_spline(time)

                except:
                    a = [(v[i] - v[i-1]) / (time[i] - time[i-1] + 1e-08)
                         for i in range(1, len(v))]
                    a += [a[-1]]
                    a = np.array(a)
                    a = np.convolve(a, np.hamming(smooth), mode="same")

                theta = np.arctan2(vy, v) * 180/np.pi
                theta = np.convolve(theta, np.hamming(smooth), mode="same")

                # store aggregate traj data
                all_time.append(time)
                all_v.append(v)
                all_x.append(x)
                all_y.append(y)
                all_a.append(a)
                all_theta.append(theta)
                all_lengths.append(length)
                all_ids.append(obj_idx)

        ax1.set_ylim([0, 1800])
        ax2.set_ylim(v_range)
        ax3.set_ylim(a_range)
        ax4.set_ylim(theta_range)

        ax1.set_xlim([0, 60])
        ax2.set_xlim([0, 60])
        ax3.set_xlim([0, 60])
        ax4.set_xlim([0, 60])

        ax1.set_ylabel("x-position (ft)", fontsize=20)
        ax2.set_ylabel("Velocity (ft/s)", fontsize=20)
        ax3.set_ylabel("Acceleration ($ft/s^2$)", fontsize=20)
        ax4.set_ylabel("Heading angle (deg)", fontsize=20)

        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax4.tick_params(axis='both', which='major', labelsize=14)

        if a_idx == len(anns) - 1:
            ax1.set_xlabel("Time (s)", fontsize=20)
            ax2.set_xlabel("Time (s)", fontsize=20)
            ax3.set_xlabel("Time (s)", fontsize=20)
            ax4.set_xlabel("Time (s)", fontsize=20)
        # else:
        #     ax1.set_xticks([])
        #     ax2.set_xticks([])
        #     ax3.set_xticks([])
        #     ax4.set_xticks([])

        if a_idx == 0:
            ax1.set_title("a.) All Vehicle Trajectories", fontsize=24)
            ax2.set_title("b.) Selected Velocity", fontsize=24)
            ax3.set_title("c.) Selected Acceleration", fontsize=24)
            ax4.set_title("d.) Selected Heading Angle", fontsize=24)

        gcolor = np.array([0, 0, 1])
        vcolor = np.array([1, 0, 0])
        acolor = np.array([1, 0.4, 0])
        tcolor = np.array([1, 1, 0])

        for i in range(len(all_x)):
            all_time[i] -= t0

            i_color = np.random.rand(3)*0.5 + 0.5
            i_color[2] = 0
            i_color[1] = i_color[1] - i_color[1]*0.2
            i_color = [0, 0, 0]
            if i == selected_idx:
                i_color = gcolor

            # plot single trajectory
            # else:
            #     continue

            # plot velocity
            points = np.array([all_time[i], all_v[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            vmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < v_range[1], np.min(
                segments[:, :, 1], axis=1) > v_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments), 3]) * vcolor
                clr[vmask] = gcolor
                lc = LineCollection(segments, colors=clr)
                ax2.add_collection(lc)

            # plot acceleration
            points = np.array([all_time[i], all_a[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            amask = np.logical_and(np.max(segments[:, :, 1], axis=1) < a_range[1], np.min(
                segments[:, :, 1], axis=1) > a_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments), 3]) * acolor
                clr[amask] = gcolor
                lc = LineCollection(segments, colors=clr)
                ax3.add_collection(lc)

            # plot theta
            points = np.array([all_time[i], all_theta[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            tmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < theta_range[1], np.min(
                segments[:, :, 1], axis=1) > theta_range[0])
            if i == selected_idx:
                clr = np.ones([len(segments), 3]) * tcolor
                clr[tmask] = gcolor
                lc = LineCollection(segments, colors=clr)
                ax4.add_collection(lc)

            # ax1.plot(all_time[selected_idx],all_x[selected_idx],color = [0,0,0],linewidth = 3)
            # ax1.plot(all_time[selected_idx],all_x[selected_idx],color = [0,0,1],linewidth = 1)

            # plot position
            lw = 4
            # if i == selected_idx:
            #     lw = 5
            points = np.array([all_time[i], all_x[i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            clr = np.ones([len(segments), 3]) * i_color
            clr[np.invert(amask)] = acolor
            clr[np.invert(tmask)] = tcolor
            clr[np.invert(vmask)] = vcolor

            lc = LineCollection(segments, colors=clr, linewidth=lw)
            ax1.add_collection(lc)

            # plot radar data
            # keep = []
            # for j in range(len(all_x[i])):
            #     if all_x[i][j] % 300 < 1:
            #         keep.append(j)
            # tsub = all_time[i][keep]
            # xsub = all_x[i][keep]
            # vsub = all_v[i][keep]
            # colors = np.zeros([len(vsub),3])
            # colors [:,0] = 1 - vsub/30
            # colors [:,1] = 0 + vsub/60
            # colors = np.clip(colors,0,1)
            # ax1.scatter(tsub,xsub, color = colors)

    plt.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig("trajectories.pdf", bbox_inches="tight")
    # for i in range(len(all_x)):
    #         all_time[i] -= t0
    #         color = np.random.rand(3)*0.5 + 0.5
    #         color[2] = 0
    #         color[1] = color[1] - color[1]*0.2

    #         # axs[0].scatter(all_time[i],all_x[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
    #         # axs[1].scatter(all_time[i],all_v[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)
    #         # axs[2].scatter(all_time[i],all_y[i],c = colors[cidx:cidx+1]/(i%3+1),marker = mk)

    #         ax1.plot(all_time[i],all_x[i],color = color)#/(i%1+1))
    #         #all_x2 = [all_lengths[i] + item for item in all_x[i]]
    #         #ax1.fill_between(all_time[i],all_x[i],all_x2,color = color)

    #ax2.plot(all_time[selected_idx],all_v[selected_idx],color = [0,0,1])
    #ax3.scatter(all_time[selected_idx],all_a[selected_idx],color = [0,0,1])
    #ax4.scatter(all_time[selected_idx],all_theta[selected_idx],color = [0,0,1])

    plt.show()


def calculate_total_variation(ann):
    x_var = []
    x_range = []

    for obj_idx in range(ann.get_unused_id()):
        x = []
        time = []

        for cam_idx, camera in enumerate(ann.cameras):
            cam_name = camera.name
            for frame in range(0, len(ann.data)):
                key = "{}_{}".format(cam_name, obj_idx)
                item = ann.data[frame].get(key)
                if item is not None:
                    x.append(ann.safe(item["x"]))
                    time.append(ann.safe(item["timestamp"]))

        if len(time) == 0:
            continue

        # sort by time
        x = np.array(x)
        time = np.array(time)  # - t0

        order = np.argsort(time)

        x = x[order]
        time = time[order]

        keep = [0]
        for i in range(1, len(time)):
            if time[i] >= time[i-1] + 0.01:
                keep.append(i)

        x = x[keep]
        ran = (max(x) - min(x))
        variation = sum([(np.abs(x[i] - x[i-1]) if np.abs(x[i] -
                        x[i-1]) > 0.5 else 0) for i in range(1, len(x))])

        x_range.append(ran)
        x_var.append(variation)

    print("Total vs True variation: {}/{}   ({}x)".format(sum(x_var),
          sum(x_range), sum(x_var)/sum(x_range)))

    return x_var, x_range


def calculate_feasibility(ann):
    v_range = [0, 140]
    a_range = [-20, 20]
    theta_range = [-25, 25]

    f_percentage = []  # one value per object
    for obj_idx in range(ann.get_unused_id()):
        x = []
        y = []
        time = []

        for cam_idx, camera in enumerate(ann.cameras):
            cam_name = camera.name
            for frame in range(0, len(ann.data)):
                key = "{}_{}".format(cam_name, obj_idx)
                item = ann.data[frame].get(key)
                if item is not None:
                    x.append(ann.safe(item["x"]))
                    y.append(ann.safe(item["y"]))
                    time.append(ann.safe(item["timestamp"]))
                    length = item["l"]

        if len(time) == 0:
            continue

        # sort by time
        x = np.array(x)
        y = np.array(y)
        time = np.array(time)  # - t0

        order = np.argsort(time)

        x = x[order]
        y = y[order]
        time = time[order]

        keep = [0]
        for i in range(1, len(time)):
            if time[i] >= time[i-1] + 0.01:
                keep.append(i)

        x = x[keep]
        y = y[keep]
        time = time[keep]

        # estimate derivative qualities
        if len(time) > 1:

            # finite difference velocity estimation
            v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08)
                 for i in range(1, len(x))]
            v += [v[-1]]
            v = np.array(v)
            #v = np.convolve(v,np.hamming(smooth),mode = "same")
            v *= -1

            vy = [(y[i] - y[i-1]) / (time[i] - time[i-1] + 1e-08)
                  for i in range(1, len(y))]
            vy += [vy[-1]]
            vy = np.array(vy)

            a = [(v[i] - v[i-1]) / (time[i] - time[i-1] + 1e-08)
                 for i in range(1, len(v))]
            a += [a[-1]]
            a = np.array(a)

            theta = np.arctan2(vy, v) * 180/np.pi

            points = np.array([time, v]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            vmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < v_range[1], np.min(
                segments[:, :, 1], axis=1) > v_range[0])

            points = np.array([time, a]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            amask = np.logical_and(np.max(segments[:, :, 1], axis=1) < a_range[1], np.min(
                segments[:, :, 1], axis=1) > a_range[0])

            points = np.array([time, theta]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            tmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < theta_range[1], np.min(
                segments[:, :, 1], axis=1) > theta_range[0])

            total_mask = np.logical_and(tmask, vmask, amask)

            percent_feasible = np.sum(
                total_mask.astype(int)) / total_mask.shape[0]
            f_percentage.append(percent_feasible)
    return f_percentage


def calculate_feasibility_spline(ann):
    v_range = [0, 140]
    a_range = [-20, 20]
    theta_range = [-25, 25]

    f_percentage = []  # one value per object
    for obj_idx in range(ann.get_unused_id()):
        x = []
        y = []
        time = []

        for cam_idx, camera in enumerate(ann.cameras):
            cam_name = camera.name
            for frame in range(0, len(ann.data)):
                key = "{}_{}".format(cam_name, obj_idx)
                item = ann.data[frame].get(key)
                if item is not None:
                    x.append(ann.safe(item["x"]))
                    y.append(ann.safe(item["y"]))
                    time.append(ann.safe(item["timestamp"]))
                    length = item["l"]

        if len(time) == 0:
            continue

        # sort by time
        x = np.array(x)
        y = np.array(y)
        time = np.array(time)  # - t0

        order = np.argsort(time)

        x = x[order]
        y = y[order]
        time = time[order]

        spl_x, spl_y = ann.splines[obj_idx]

        if spl_x is None or spl_y is None:
            continue
        spl_dx = spl_x.derivative()
        spl_ddx = spl_dx.derivative()
        spl_dy = spl_y.derivative()

        v = spl_dx(time)
        a = spl_ddx(time)
        vy = spl_dy(time)
        theta = np.arctan2(vy, v) * 180/np.pi

        keep = [0]
        for i in range(1, len(time)):
            if time[i] >= time[i-1] + 0.01:
                keep.append(i)

        x = x[keep]
        y = y[keep]
        time = time[keep]

        # # estimate derivative qualities
        # if len(time) > 1:

        #     # finite difference velocity estimation
        #     v = [(x[i] - x[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(x))]
        #     v += [v[-1]]
        #     v  = np.array(v)
        #     #v = np.convolve(v,np.hamming(smooth),mode = "same")
        #     v *= -1

        #     vy = [(y[i] - y[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(y))]
        #     vy += [vy[-1]]
        #     vy = np.array(vy)

        #     a = [(v[i] - v[i-1]) / (time[i] - time[i-1] + 1e-08) for i in range(1,len(v))]
        #     a += [a[-1]]
        #     a = np.array(a)

        #     theta = np.arctan2(vy,v) * 180/np.pi

        points = np.array([time, v]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        vmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < v_range[1], np.min(
            segments[:, :, 1], axis=1) > v_range[0])

        points = np.array([time, a]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        amask = np.logical_and(np.max(segments[:, :, 1], axis=1) < a_range[1], np.min(
            segments[:, :, 1], axis=1) > a_range[0])

        points = np.array([time, theta]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        tmask = np.logical_and(np.max(segments[:, :, 1], axis=1) < theta_range[1], np.min(
            segments[:, :, 1], axis=1) > theta_range[0])

        total_mask = np.logical_and(tmask, vmask, amask)

        percent_feasible = np.sum(total_mask.astype(int)) / total_mask.shape[0]
        f_percentage.append(percent_feasible)

    return f_percentage


def annotator_rmse(ann):
    mses = []

    i = 0
    obj_annotations = []
    cam = None
    while i < ann.get_unused_id():
        for frame_data in ann.data:
            for obj in frame_data.values():
                if obj["id"] == i:
                    obj_annotations.append(obj)
                    cam = obj["camera"]
        if i % 5 == 4:
            boxes = torch.stack([torch.tensor([obj["x"], obj["y"], obj["l"], obj["w"],
                                obj["h"], obj["direction"]]).float() for obj in obj_annotations])
            im_boxes = ann.hg.state_to_im(boxes, name=cam)

            im_pos = torch.mean(im_boxes[:, [2, 3], :], dim=1)
            mean = torch.mean(im_pos, dim=0)

            rmse = (((im_pos[:, 0] - mean[0]).pow(2) + (im_pos[:, 1] -
                    mean[1]).pow(2)).sum(dim=0)/im_pos.shape[0]).sqrt()
            mses.append(rmse)

            obj_annotations = []
        i += 1

    rmse = torch.sqrt(sum(mses)/len(mses))
    print("RMSE: {}".format(rmse))


def plot_deltas(ann, cam="p1c5"):

    all_ts = []
    for frame in ann.data:
        for item in frame:
            if cam in item:
                all_ts.append(frame[item]["timestamp"])
                break

    deltas = [all_ts[i] - all_ts[i-1] for i in range(1, len(all_ts))]
    plt.plot(deltas)
    plt.ylim([-0.01, 0.05])
    plt.xlim([0, 250])
    plt.show()





#%%
if __name__ == "__main__":
        
    from i24_rcs import I24_RCS
    rcs  = I24_RCS("test.cpkl",aerial_ref_dir="/home/worklab/Documents/datasets/I24-3D/rcs_1/aerial_ref_1",im_ref_dir="/home/worklab/Documents/datasets/I24-3D/rcs_1/cam_ref_1",downsample = 1,default = "reference",MC3D = True)
    
    video_dir = "/home/worklab/Documents/datasets/I24-3D/video"
    data_dir  = "/home/worklab/Documents/datasets/I24-3D/data"
    
    ann = Scene(video_dir,data_dir,scene_id = 1,start_frame = 2200)
    ann.correct_curve()
    ann.fill_buffer(50)    
    ann.run()
