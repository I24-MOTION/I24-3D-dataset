

# load v3 homography
from i24_rcs import I24_RCS

import _pickle as pickle
import time



import torch
import re
import cv2 as cv
import string
import copy
import cv2
import os
import numpy as np
import warnings
import random
random.seed(1)

warnings.filterwarnings("ignore")


import torch.multiprocessing as mp
ctx = mp.get_context('spawn')

# filter and CNNs


class Frame_Labeler:
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
         for labels later down the line
    """
    
        
    def __init__(self, im_dir, lab_dir,mask_dir = None,advance = True):
        self.label_dir = lab_dir


        self.rcs = I24_RCS("/home/worklab/Documents/datasets/more_3D_frames/hg_67e62dfd1ffd2fe3d61ee2d0.cpkl",default = "reference",downsample = 2)
        to_add = []
        for key in self.rcs.correspondence.keys():
            if "CO4" in key or "C03" in key:
                direction = key.split("_")[1]
                split_key = key.split("_")[0]
                if direction == "WB":
                    if split_key + "_EB" not in self.rcs.correspondence.keys():
                        to_add.append([split_key + "_EB", self.rcs.correspondence[key]])
                elif direction == "EB":
                     if split_key + "_WB" not in self.rcs.correspondence.keys():
                         to_add.append([split_key + "_WB", self.rcs.correspondence[key]])
        for item in to_add:
            self.rcs.correspondence[item[0]] = item[1]
            
        self.classes = {

                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi_sleeper",
                    5:"semi_sleeper_double",
                    6:"semi_nonsleeper",
                    7:"semi_nonsleeper_double",
                    8:"truck_box",
                    9:"truck_flatbed",
                    10:"truck_dump",
                    11:"truck_rv",
                    12:"truck_pickup_chassis", # but not flatbed
                    13:"truck_lawn",
                    14:"truck_ambulance",
                    15:"truck", # other
                    16:"motorcycle",
                    17:"bus",
                    18:"bus_school",
                    }
        classes2 = {}
        for key in self.classes.keys():
            classes2[self.classes[key]] = key
        for key in classes2:
            self.classes[key] = classes2[key]
     
        self.class_dims = {
                "sedan":[16,6,4],
                "midsize":[183/12,72/12,67/12],
                "van":[19,6,7.5],
                "pickup":[19,72/12,67/12],
                "semi_sleeper":[70,9,12],
                "semi_nonsleeper":[70,9,12],
                "truck_box":[30,9,12],
                "truck": [25,9,12],
                "motorcycle":[7,3,4],
                "bus":[30,9,12],
            }
        
     
        self.cont = True
        self.new = None
        self.clicked = False
        self.active_command = "ADD"
        self.right_click = False
        self.copied_idx = None

       
        self.MASK = False
        self.TEXT = True 
        self.PLOT = True
        self.DOUBLE_FOR_NEXT = False
        
        self.mask_ims = {}
        if mask_dir is not None:
            mask_paths = os.listdir(mask_dir)
            for path in mask_paths:
                if "1080" in path:
                    key = path.split("_")[0]
                    path = os.path.join(mask_dir, path)
                    im = cv2.imread(path)
    
                    self.mask_ims[key] = im
                    
        self.im_list = os.listdir(im_dir)
        self.im_list = [os.path.join(im_dir,file) for file in self.im_list]            
        
        for i  in range(len(self.im_list)):
            self.im_list[i] = self.im_list[i][::-1]
        self.im_list.sort()
        for i  in range(len(self.im_list)):
           self.im_list[i] = self.im_list[i][::-1]
            
        #self.im_list.sort()        
        #random.shuffle(self.im_list)

        self.im_list_idx = 0
        self.load_im()
        
        if advance:
            self.advance_to_unlabeled()
    
        
    def export(self,destination_dir = "/home/worklab/Documents/datasets/more_3D_frames/label_export",mask_dir = "/home/worklab/Documents/datasets/more_3D_frames/mask"):
        
        # for each frame
        for i in range(len(self.im_list)):
            im_path  = self.im_list[i]
            self.cam =   re.search("P\d\dC\d\d",im_path).group(0)

            # load labels if we can
            base_name = im_path.split("/")[-1].split(".")[0]
            label_name = base_name + ".cpkl"
            label_path =  os.path.join(self.label_dir,label_name)
            
            # if there is a label file load
            if os.path.exists(label_path):
                with open(label_path,"rb") as f:
                    self.label = pickle.load(f)
            else:
                continue
            
            # append im_box and box
            ts_data = self.label
            # if len(ts_data) > 0:
            #     boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
                
            #     # convert into image space
            #     im_boxes = self.rcs.state_to_im(boxes,name = [self.cam for _ in self.label])
                 
            # for i in range (len(ts_data)):
            #     # for each object, append the image-space box to the ts_data annotation
            #     self.label[i]["im_box"] = im_boxes[i]
            #     self.label[i]["box"] = boxes[i]
            #     self.label[i]["id"] = i
            # # write to new location
            
            if len(ts_data) > 0:
                if False: #--------------------------------------------------------------------------------------------------------------------------------------------- single or double box for trailers
                    boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() for obj in ts_data])
                    # convert into image space
                    im_boxes = self.rcs.state_to_im(boxes,name = [self.cam for _ in self.label])
                    
                    for i in range (len(ts_data)):
                        # for each object, append the image-space box to the ts_data annotation
                        ts_data[i]["im_box"] = im_boxes[i]
                        ts_data[i]["box"] = boxes[i]
                        if "gen" not in ts_data[i].keys(): ts_data[i]["gen"] = "Manual"
                
                else:
                    # separate cab and trailer boxes
                    trailers = []
                    for obj in ts_data:
                        if obj["cab_length"] > 0:
                            item = obj
                            front = torch.tensor([item["x"]+item["l"]*item["direction"] - item["cab_length"]*item["direction"],item["y"],item["cab_length"],item["w"],item["h"],item["direction"]]).float()
                            back =  torch.tensor([item["x"],item["y"],item["l"] - item["cab_length"],item["w"],item["h"],item["direction"]]).float()
                            front_im = self.rcs.state_to_im(front.unsqueeze(0),name = [self.cam])
                            back_im =  self.rcs.state_to_im(back.unsqueeze(0),name = [self.cam])
                            
                            obj_new = copy.deepcopy(obj)
                            
                            obj["box"] = front
                            obj["im_box"] = front_im
                            
                            obj_new["box"] = back
                            obj_new["im_box"] = back_im
                            
                            if "semi" in obj["class"]: # front = original, back = trailer_s
                                obj_new["class"] = "trailer_sem"
                            else:   # front = orighinal, back = trailer
                                obj_new["class"] = "trailer"

                            trailers.append(obj_new)
                            
                        else:
                            box = torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"]]).float() 
                            im_box = self.rcs.state_to_im(box.unsqueeze(0),name = [self.cam])
                            obj["box"] = box
                            obj["im_box"] = im_box

                    ts_data += trailers
                    
                    
                    
                    
            
            with open(os.path.join(destination_dir,label_name),"wb") as f:
                  pickle.dump(ts_data,f)
              
            
            # ensure that there is a mask image written
            mask_path = os.path.join(mask_dir,self.cam + ".png")
            if not os.path.exists(mask_path):
                mask_poly = []
                try:
                    if len(self.rcs.correspondence[self.cam+"_EB"]["mask"]) > 0:
                        mask_poly = self.rcs.correspondence[self.cam+"_EB"]["mask"]
                except:
                    pass
                try:
                    if len(self.rcs.correspondence[self.cam+"_WB"]["mask"]) > 0:
                        mask_poly = self.rcs.correspondence[self.cam+"_WB"]["mask"]
                except:
                    pass

                # default = no mask                
                mask = (np.ones(self.im.shape)) * 255
                
                if len(mask_poly) > 2:
                    neg_mask = mask * 0
                    mask_poly = (np.stack([np.array(pt) for pt in mask_poly]).reshape(
                        1, -1, 2)*0.5).astype(np.int32) # *0.5 for 1080p
                    mask = cv2.fillPoly(
                        neg_mask, mask_poly,  (255,255,255), lineType=cv2.LINE_AA)
                cv2.imwrite(mask_path,mask)
        
        # reset at the end
        self.im_list_idx = 0
        self.load_im()
        
    
    def advance_to_unlabeled(self):
        # skip to first unlabeled frame
        while True:
            self.im_list_idx += 1
            
            im_path  = self.im_list[self.im_list_idx]
            base_name = im_path.split("/")[-1].split(".")[0]
            label_name = base_name + ".cpkl"
            label_path =  os.path.join(self.label_dir,label_name)
            if not os.path.exists(label_path):
                break
            
        self.im_list_idx -= 1
            
        self.load_im()
        self.plot()
        
    def load_im(self):
        #load image
        im_path  = self.im_list[self.im_list_idx]
        self.im = cv2.imread(im_path)
        self.cam =   re.search("P\d\dC\d\d",im_path).group(0)

        # load labels if we can
        base_name = im_path.split("/")[-1].split(".")[0]
        label_name = base_name + ".cpkl"
        label_path =  os.path.join(self.label_dir,label_name)
        if os.path.exists(label_path):
            with open(label_path,"rb") as f:
                self.label = pickle.load(f)
        else:
            self.label = []
            
        # generate mask image
        self.mask = None
        mask_poly = []
        try:
            if len(self.rcs.correspondence[self.cam+"_EB"]["mask"]) > 0:
                mask_poly = self.rcs.correspondence[self.cam+"_EB"]["mask"]
        except:
            pass
        try:
            if len(self.rcs.correspondence[self.cam+"_WB"]["mask"]) > 0:
                mask_poly = self.rcs.correspondence[self.cam+"_WB"]["mask"]
        except:
            pass
        
            
        if len(mask_poly) > 2:
            mask_poly = (np.stack([np.array(pt) for pt in mask_poly]).reshape(
                1, -1, 2)*0.5).astype(np.int32) # *0.5 for 1080p
            mask = (np.ones(self.im.shape)) * 0.4
            mask = cv2.fillPoly(
                mask, mask_poly,  (1,1,1), lineType=cv2.LINE_AA)
            self.mask = mask
            
            
            
    def save(self):
        im_path  = self.im_list[self.im_list_idx]
        base_name = im_path.split("/")[-1].split(".")[0]
        label_name = base_name + ".cpkl"
        label_path =  os.path.join(self.label_dir,label_name)
        
        with open(label_path,"wb") as f:
            pickle.dump(self.label,f)
            
        print("Saved {}".format(label_path))
        
    def safe(self, x):
        """
        Casts single-element tensor as an variable, otherwise does nothing
        """
        try:
            x = x.item()
        except:
            pass
        return x

  
    def plot(self):
        self.plot_frame = self.im.copy()
        if self.PLOT:
            # stack labels into tensor
            boxes = [torch.tensor([item["x"],item["y"],item["l"],item["w"],item["h"],item["direction"]]) for item in self.label]
            if len(boxes) > 0:
                boxes = torch.stack(boxes)
            
                #stack label names
                labels = None
                if self.TEXT:
                    labels = ["{} {}".format(self.label[idx]["class"],idx) for idx in range(len(self.label))]
                
                self.rcs.plot_state_boxes(self.plot_frame, boxes,times = None, thickness = 2, labels = labels,name = [self.cam for _ in self.label],size = 0.8)
            
            trucks = []
            for item in self.label:
                if item["cab_length"] > 0:
                    trucks.append(torch.tensor([item["x"]+item["l"]*item["direction"] - item["cab_length"]*item["direction"],item["y"],item["cab_length"],item["w"],item["h"],item["direction"]]))
            if len(trucks) > 0:
                boxes = torch.stack(trucks)
                self.rcs.plot_state_boxes(self.plot_frame, boxes,times = None, thickness = 2, name = [self.cam for _ in trucks],color = (0,100,255))
              
            if self.mask is not None and self.MASK:
                self.plot_frame = (self.mask * self.plot_frame.astype(np.float64)).astype(np.uint8)
            
    def add(self,location):
        
        
        xy = self.box_to_state(location)[0,:].data.numpy()
        
        # create new object
        # 2022 nissan rogue dimensions : 183″ L x 72″ W x 67″ H
        obj = {
            "l": 16,
            "w": 6,
            "h": 4,
            "class":"sedan",
            "cab_length": 0,
            "x":xy[0],
            "y":xy[1],
            "direction": np.sign(xy[1])
            }
        
        
        
        self.label.append(obj)
        
        self.copied_idx = len(self.label)-1
        self.active_command = "COPY PASTE"
            

    def copy_paste(self,point):     
        if self.copied_idx is None:
           
            # assign nearest object to click as copied_idx
            obj_idx = self.find_box(point)
            if obj_idx is None:
                return
            else:
                self.copied_idx = obj_idx
            
            
            
        
        else: # paste the copied box
            start = time.time()
            state_point = self.box_to_state(point)[0]

            dx = state_point[0] - self.label[self.copied_idx]["x"]
            dy = state_point[1] - self.label[self.copied_idx]["y"]
          
            self.label[self.copied_idx]["x"] += dx
            self.label[self.copied_idx]["y"] += dy
            
            
    def dimension(self,obj_idx,box):
        """
        Adjust relevant dimension in all frames based on input box. Relevant dimension
        is selected based on:
            1. if self.right_click, height is adjusted - in this case, a set ratio
               of pixels to height is used because there is inherent uncertainty 
               in pixels to height conversion
            2. otherwise, object is adjusted in the principle direction of displacement vector
        """
        if obj_idx is None:
            return
        
        state_box = self.box_to_state(box)
        dx = state_box[1,0] - state_box[0,0]
        dy = state_box[1,1] - state_box[0,1]
        
        if np.abs(dx) > np.abs(dy): 
            self.label[obj_idx]["l"] += dx
        else:
            self.label[obj_idx]["w"] += dy
                
                
    def box_to_state(self,point,direction = False):
        """
        Input box is a 2D rectangle in image space. Returns the corresponding 
        start and end locations in space
        point - indexable data type with 4 values (start x/y, end x/y)
        state_point - 2x2 tensor of start and end point in space
        """
        point = point.copy()

        point1 = torch.tensor([point[0],point[1]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point2 = torch.tensor([point[2],point[3]]).unsqueeze(0).unsqueeze(0).repeat(1,8,1)
        point = torch.cat((point1,point2),dim = 0)
        
        state_point = self.rcs.im_to_state(point,name = [self.cam,self.cam], heights = torch.tensor([0,0]))    
        return state_point[:,:2]
    
    
    def on_mouse(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN and not self.clicked:
            self.start_point = (x, y)
            self.clicked = True
        elif event == cv.EVENT_LBUTTONUP:
            box = np.array([self.start_point[0], self.start_point[1], x, y])
            self.new = box
            self.clicked = False

        # some commands have right-click-specific toggling
        elif event == cv.EVENT_RBUTTONDOWN:
            self.right_click = not self.right_click
            self.copied_idx = None

        # elif event == cv.EVENT_MOUSEWHEEL:
        #      print(x,y,flags)

    def find_box(self, point):
        point = point.copy()

        
        point = torch.tensor([point[0], point[1]]).unsqueeze(
            0).unsqueeze(0).repeat(1, 8, 1)
        state_point = self.rcs.im_to_state(
            point, name=[self.cam], heights=torch.tensor([0])).squeeze(0)

        print(point,state_point)
        min_dist = np.inf
        min_id = None
        for bidx,box in enumerate(self.label):

            dist = (box["x"] - state_point[0])**2 + \
                (box["y"] - state_point[1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = bidx

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
        self.save()
        cv2.destroyAllWindows()
        

    def undo(self):
        if self.label_buffer is not None:
            self.label = self.label_buffer
            self.label_buffer = None
            self.plot()
        else:
            print("Can't undo")


    def delete(self,obj_idx):
        del self.label[obj_idx]



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
                self.label_buffer = copy.deepcopy(self.label)

                # Add and delete objects
                if self.active_command == "DELETE":
                    obj_idx = self.find_box(self.new)
                    self.delete(obj_idx)

                elif self.active_command == "ADD":
                    # get obj_idx
                    self.add(self.new)
                    self.DOUBLE_FOR_NEXT = False
                    
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


                # correct vehicle class
                elif self.active_command == "VEHICLE CLASS":
                    obj_idx = self.find_box(self.new)
                    try:
                        cls = (self.keyboard_input())
                    except:
                        cls = "midsize"
                    self.change_class(obj_idx, cls)

              


                self.plot()
                self.new = None

            # Show frame
            cv2.imshow("window", self.plot_frame)
            title = "{} {}     {}/{}: {} ---- {}".format("R" if self.right_click else "", self.active_command,self.im_list_idx,len(self.im_list),self.im_list[self.im_list_idx].split("/")[-1],self.copied_idx)
            cv2.setWindowTitle("window", str(title))



            # Handle keystrokes
            key = cv2.waitKey(1)


          
            
            if key == ord('e'):
                if self.DOUBLE_FOR_NEXT:
                    self.save()
                    self.im_list_idx += 1
                    self.load_im()
                    self.plot()
                    self.DOUBLE_FOR_NEXT = False
                else:
                    self.DOUBLE_FOR_NEXT = True
                    
           
            elif key == ord("r"):
                self.save()
                
                self.advance_to_unlabeled()
                        
                
            elif key == ord("q"):
                self.save()
                cv2.destroyAllWindows()
                break
            
            elif key == ord("w"):
                self.save()
           
            elif key == ord("u"):
                self.undo()
        

            elif key == ord("t"):
                self.TEXT = not self.TEXT
                self.plot()

            elif key == ord("m"):
                self.MASK = not self.MASK
                self.plot()
            elif key == ord("y"):
                self.PLOT = not self.PLOT
                self.plot()

            elif key == ord(" ") and self.copied_idx is not None:
                #change class
                class_idx = self.classes[self.label[self.copied_idx]["class"]]
                class_idx += 1
                try:
                    new_class = self.classes[class_idx]
                except KeyError:
                    class_idx = 0
                    new_class = self.classes[class_idx]
                    time.sleep(0.5)
                self.label[self.copied_idx]["class"] = new_class
                
                
                if new_class in self.class_dims.keys():
                    self.label[self.copied_idx]["l"] = self.class_dims[new_class][0]
                    self.label[self.copied_idx]["w"] = self.class_dims[new_class][1]
                    self.label[self.copied_idx]["h"] = self.class_dims[new_class][2]
                    
                self.plot()
                
                
            elif key == ord("~"):
                os.remove(self.im_list[self.im_list_idx])
                print("Removed {}".format(self.im_list[self.im_list_idx]))
                del self.im_list[self.im_list_idx]
                self.load_im()
                self.plot()

            # toggle commands
            elif key == ord("a"):
                self.active_command = "ADD"
            elif key == ord("x"):
                self.active_command = "DELETE"
            elif key == ord("d"):
                self.active_command = "DIMENSION"
            elif key == ord("c"):
                self.active_command = "COPY PASTE"
            elif key == ord("v"):
                self.active_command = "CAB LENGTH"

            elif self.active_command == "COPY PASTE" and self.copied_idx is not None:
                nudge = 0.5
                if "C05" in self.cam or "C06" in self.cam:
                    nudge *= -1
                
                
                if key == ord("1") or int(key) == 177:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "x"
                    else:
                        k = "y"
                        
                    self.label[self.copied_idx][k] -= nudge
                    self.plot()
                if key == ord("3") or int(key) == 179:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "x"
                    else:
                        k = "y"
                        
                    self.label[self.copied_idx][k] += nudge
                    self.plot()
                    
                if key == ord("5") or int(key) == 181:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "y"
                    else:
                        k = "x"
                        
                    self.label[self.copied_idx][k] += nudge
                    self.plot()
                if key == ord("2") or int(key) == 178:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "y"
                    else:
                        k = "x"
                        
                    self.label[self.copied_idx][k] -= nudge
                    self.plot()

            elif self.active_command == "DIMENSION" and self.copied_idx is not None:
                nudge = 0.25
                if "C05" in self.cam or "C06" in self.cam:
                    nudge *= -1
                    
                if key == ord("1") or key == 177: # 149 is the numpad '1' for me, you may need to change these
                
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "l"
                    else:
                        k = "w"
                    self.label[self.copied_idx][k] += nudge
                    self.plot()
                if key == ord("3") or key == 179:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "l"
                    else:
                        k = "w"
                    self.label[self.copied_idx][k] -= nudge
                    self.plot()    
                
                if key == ord("5") or key == 181:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "w"
                    else:
                        k = "l"
                    self.label[self.copied_idx][k] -= nudge
                    self.plot()

                if key == ord("2") or key == 178:
                    if "C04" in self.cam or "C03" in self.cam:
                        k = "w"
                    else:
                        k = "l"
                    self.label[self.copied_idx][k] += nudge
                    self.plot()
                    
                if key == ord("7") or key == 179:
                    self.label[self.copied_idx]["h"] -= 0.25
                    self.plot()
                if key == ord("8") or key == 178:
                    self.label[self.copied_idx]["h"] += 0.25
                    self.plot()
                if key == ord("0") or key == 181:
                    self.label[self.copied_idx]["cab_length"] -= np.abs(nudge)
                    if self.label[self.copied_idx]["cab_length"] < 0:
                        self.label[self.copied_idx]["cab_length"] = 0
                    self.plot()
                if key == ord(".") or key == 178:
                    self.label[self.copied_idx]["cab_length"] += np.abs(nudge)
                    self.plot()    
                









#%%
if __name__ == "__main__":
          
    im_dir = "/home/worklab/Documents/datasets/more_3D_frames/im"
    lab_dir = "/home/worklab/Documents/datasets/more_3D_frames/label"
    
    f = Frame_Labeler(im_dir, lab_dir,advance = False)
    f.run()
    #f.export()