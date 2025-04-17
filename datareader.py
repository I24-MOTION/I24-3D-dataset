"""
Datareader essentially provides a tool for plotting .csv 3D tracking data files
that correspond to multiple video sequences/cameras. This task introduces the 
complexity of time synchronization relative to plotting tracking results for a
single camera. Additionally, functionality is provided for converting the 
multiple-camera .csv data file into the corresponding data for a single camera.
Out-of-bounds objects are removed from the dataset and all objects are shifted into
phase with that camera's timestamps
"""

import csv
import re
import numpy as np
import cv2
import torch

import PyNvCodec as nvc
import PytorchNvCodec as pnvc



def load_queue_vpf(q,sequence,gpuID,buffer_size,resize,start_frame):
        print("device id: {}".format(gpuID))

        resize = (resize[1],resize[0])
        # device = torch.cuda.device("cuda:{}".format(gpuID))
        file = sequence
        
        # initialize Decoder object
        nvDec = nvc.PyNvDecoder(file, gpuID)
        target_h, target_w = nvDec.Height(), nvDec.Width()
    
        to_rgb = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpuID)
        to_planar = nvc.PySurfaceConverter(nvDec.Width(), nvDec.Height(), nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpuID)
    
    
        cspace, crange = nvDec.ColorSpace(), nvDec.ColorRange()
        if nvc.ColorSpace.UNSPEC == cspace:
            cspace = nvc.ColorSpace.BT_601
        if nvc.ColorRange.UDEF == crange:
            crange = nvc.ColorRange.MPEG
        cc_ctx = nvc.ColorspaceConversionContext(cspace, crange)
        
        
        # get frames from one file
        frame_counter = -1
        while True:
            if q.qsize() < buffer_size:
                
                
                # Obtain NV12 decoded surface from decoder;
                raw_surface = nvDec.DecodeSingleSurface()#pkt)
                if raw_surface.Empty():
                    break
                
                # used to skip to frame 
                frame_counter += 1
                if frame_counter < start_frame:
                    continue
    
                # Convert to RGB interleaved;
                rgb_byte = to_rgb.Execute(raw_surface, cc_ctx)
            
                # Convert to RGB planar because that's what to_tensor + normalize are doing;
                rgb_planar = to_planar.Execute(rgb_byte, cc_ctx)
    

            
                # likewise, end of video file
                if rgb_planar.Empty():
                    break
                
                rgb_planar = rgb_planar.Clone(gpuID)
                
                # del rgb_byte
                
                # Create torch tensor from it and reshape because
                # pnvc.makefromDevicePtrUint8 creates just a chunk of CUDA memory
                # and then copies data from plane pointer to allocated chunk;
                surfPlane = rgb_planar.PlanePtr()
                # del rgb_planar
                
                surface_tensor = pnvc.makefromDevicePtrUint8(surfPlane.GpuMem(), surfPlane.Width(), surfPlane.Height(), surfPlane.Pitch(), surfPlane.ElemSize())
                del surfPlane
                
                surface_tensor.resize_(3, target_h,target_w)
                
                try:
                    surface_tensor = torch.nn.functional.interpolate(surface_tensor.unsqueeze(0),resize).squeeze(0)
                except:
                    raise Exception("Surface tensor shape:{} --- resize shape: {}".format(surface_tensor.shape,resize))
            
                # This is optional and depends on what you NN expects to take as input
                # Normalize to range desired by NN. Originally it's 
                surface_tensor = surface_tensor.type(dtype=torch.cuda.FloatTensor)
                surface_tensor = surface_tensor.permute(1,2,0).data.cpu().numpy().astype(np.uint8)[:,:,::-1].copy()
                
                # apply normalization
                #surface_tensor = F.normalize(surface_tensor,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                frame = surface_tensor
                q.put(frame)

class Camera_Wrapper_vpf():
    """
    Small wrapper around cv2.VideoCapture object that also maintains timestamp
    """
    
    def __init__(self,sequence,ctx,ds = 2,gpu = 0,start_frame = 0):
        self.cap = cv2.VideoCapture(sequence)
        self.ds = ds
        
        #checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
        #geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
       
        
        self.frame = None
        self.name = re.search("p\dc\d",sequence).group(0)
        
        self.all_ts = []
    
        self.running_frame = None
        

        
        # create shared queue
        resize = (3840,2160)
        self.queue = ctx.Queue()
        self.worker = ctx.Process(target=load_queue_vpf, args=(self.queue,sequence,gpu,20,resize,start_frame))
        self.worker.start()   
         
    def __next__(self):
        
        try:
            frame = self.queue.get(timeout = 20)
            im = frame
            
        except:
            im = None
        
        self.frame = im
        
        if self.running_frame is None:
            self.running_frame = self.frame
        else:
            self.running_frame = 0.95*self.running_frame + 0.05*self.frame
            
    def release(self):
        self.cap.release()
        
    def __len__(self):
        return int(self.cap.get(7))    
    
    def skip(self,count):
        for i in range(count):
            next(self)
        next(self)
    
        
class Camera_Wrapper():
    """
    Small wrapper around cv2.VideoCapture object that also maintains timestamp
    """
    
    def __init__(self,sequence,ds = 2):
        self.cap = cv2.VideoCapture(sequence)
        self.ds = ds
        
        #checksum_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_pixel_checksum_6.pkl"
        #geom_path="/home/worklab/Documents/derek/I24-video-processing/I24-video-ingest/resources/timestamp_geometry_4K.pkl"
       
        
        self.frame = None
        self.name = re.search("p\dc\d",sequence).group(0)
        
        self.all_ts = []
    
        self.running_frame = None
        
    def __next__(self):
        ret,self.frame = self.cap.read()
        
        self.ts = -1
        
        self.all_ts.append(self.ts)

        
        if self.ds == 2:
            self.frame = cv2.resize(self.frame,(1920,1080))
            
        if self.running_frame is None:
            self.running_frame = self.frame
        else:
            self.running_frame = 0.95*self.running_frame + 0.05*self.frame
            
    def release(self):
        self.cap.release()
        
    def __len__(self):
        return int(self.cap.get(7))    
    
    def skip(self,count):
        for i in range(count):
            self.cap.grab()
        
        next(self)

class Data_Reader():
    
    def __init__(self,data_csv,homography,metric = False):
        """
        data_csv - a data file adhering to the data template on https://github.com/DerekGloudemans/manual-track-labeler
        homgraphy - a Homography object with correspondences for all cameras for
                    which the user wishes to plot or convert data into the local time phase of
        """
        
        self.hg = homography
        self.d_idx = 0

        
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
    
        self.classes = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer",
                    }
        
        
        # load data - we store only the 3D state, direction, object ID, class, and timestamp,
        # as this information is global and all other columns are local and reproducible from the above data
        
        
        # data structure - list of dicts of dicts, indexed first by unique timestamp, then unique object
        self.data = []
        
        data = {}
        current_timestamp = -1
        timestamp_data = {}
        with open(data_csv,"r") as f:
            read = csv.reader(f)
            HEADERS = True
            for row in read:
                #get rid of first row
                if HEADERS:
                    if len(row) > 0 and row[0] == "Frame #":
                        HEADERS = False
                        camera_names = row[45]
                        self.cameras = re.findall("(p\dc\d)",camera_names)
                    continue
        
                else:
                    try:
                        x       = float(row[39])
                        y       = float(row[40]) 
                        #theta   = float(row[41])
                        w       = float(row[42])   
                        l       = float(row[43])   
                        h       = float(row[44])
                        direc   =   int(float(row[35]))
                        vel     = float(row[38])
                        id      =   int(float(row[2]))
                        cls     =       row[3]
                        ts      =  np.round(float(row[1]),4)
                        camera  =  row[36]
                        frame   =  row[0] 
                        if camera == "":
                            camera = "p1c1" # default
                        
                        if metric:
                            y = y * 3.281
                            x = x * 3.281
                            w = w * 3.281
                            l = l * 3.281
                            h = h * 3.281
                            vel = vel * 3.281
                        
                        camera_offsets = row[45]
                        camera_offsets = camera_offsets.strip("[").strip("]").split(",")
                        
                        offsets = [float(item) for item in camera_offsets]
                        offsets = dict([(self.cameras[i],offsets[i]) for i in range(len(offsets))])
                        
                    except:
                        continue

                        
                    datum = {
                        "timestamp":ts,
                        "id":id,
                        "class":cls,
                        "x":x,
                        "y":y,
                        "l":l,
                        "w":w,
                        "h":h,
                        "direction":direc,
                        "v":vel,
                        "ts_bias":offsets,
                        "camera":camera,
                        "frame":frame
                        }
                    
                    
                    if ts in data.keys():
                        data[ts][id] = datum
                    else:
                        data[ts] = {id:datum}
                    
                    # if ts == current_timestamp:
                    #     timestamp_data[id] = datum
                    # else:
                    #     if current_timestamp != -1:
                    #         if len(timestamp_data.keys()) > 0:
                    #             data[ts] =
                    #     timestamp_data = {}
                    #     current_timestamp = ts
                        
        # now, data is a dictionary keyed by timestamps. We want it to be a list, so sort keys
        data_keys = list(data.keys())
        data_keys.sort()
        
        self.data = [data[key] for key in data_keys]
        
    def __next__(self):
        try:
            if self.d_idx < len(self.data):
                datum = self.data[self.d_idx].copy()
                ts = datum[list(datum.keys())[0]]["timestamp"]
                if self.d_idx < len(self.data) -1:   
                    next_ts = self.data[self.d_idx+1][list(self.data[self.d_idx + 1].keys())[0]]["timestamp"]
                    next_datum = self.data[self.d_idx+1].copy()
                else:
                    next_ts = None
                    next_datum = None
                    
                self.d_idx += 1
               
                return datum,ts,next_ts,next_datum         
            
            else: return None,None,None,None
            
        except:
            print(self.d_idx,self.data[self.d_idx])
            
    def plot_labels(self,im,boxes,state_boxes,classes,ids,speeds,directions,times):
        im2 = im.copy()
        for i in range(len(boxes)):
            
            if type(times) == list:
                time = times[i]
            else:
                time = times
            
            label = "{} {}:".format(classes[i],ids[i])          
            #label2 = "{:.1f}mph {}".format(speeds[i],directions[i])   
            label3 = "L: {:.1f}ft".format(state_boxes[i][2])
            label4 = "W: {:.1f}ft".format(state_boxes[i][3])
            label5 = "H: {:.1f}ft".format(state_boxes[i][4])
            label6 = "{}".format(time)
            full_label = [label,label3,label4,label5,label6]
            
            longest_label = max([item for item in full_label],key = len)
            
            text_size = 1.2
            t_size = cv2.getTextSize(longest_label, cv2.FONT_HERSHEY_PLAIN,text_size , 1)[0]

            # find minx and maxy 
            minx = torch.min(boxes[i,:,0])
            maxy = torch.max(boxes[i,:,1])
            
            c1 = (int(minx),int(maxy)) 
            c2 = int(c1[0] + t_size[0] + 10), int(c1[1] + len(full_label)*(t_size[1] +4)) 
            cv2.rectangle(im2, c1, c2,(255,255,255), -1)
            
            offset = t_size[1] + 4
            for label in full_label:
                
                c1 = c1[0],c1[1] + offset
                cv2.putText(im, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
                cv2.putText(im2, label, c1, cv2.FONT_HERSHEY_PLAIN,text_size, [0,0,0], 1)
        
        im = cv2.addWeighted(im,0.7,im2,0.3,0)
        return im
        
        
#     def plot_in(self,sequences,framerate = 10, savefile = None):
#         """
#         Plots the data in each video sequence
#         sequences - (list of str) paths to video sequences, each of which must
#                     contain a unique camera identifier such as p1c4
#         """
        
#         # initialize videoCapture object for each sequence
#         cameras = []
#         for sequence in sequences:
#             cap = Camera_Wrapper(sequence)
#             next(cap)
#             cameras.append(cap)
        
#         ts_data,ts,next_ts,_ = next(self)
        
#         if savefile is not None:
#                 size = (3840,2160)
#                 out = cv2.VideoWriter(savefile,cv2.VideoWriter_fourcc(*'MPEG'), framerate, size)
        
#         while True:
            
#             # advance frames until all cameras are within 1/30 sec of one another
#             max_time = max([cam.ts for cam in cameras])
#             for cam in cameras:
#                 try:
#                     while cam.ts + 1/60.0 < max_time:
#                         next(cam)
#                 except TypeError:
#                     next(cam)
                    
#             # advance current labels until camera timestamp is between current and next set of label timestamps
#             if next_ts is None:
#                 break
#             while max_time > next_ts:
#                 if next_ts is not None:
#                     ts_data,ts,next_ts,_ = next(self)
#                 if next_ts is None:
#                     break
            
#             for camera in cameras:
#                 cam_ts = camera.ts
                
#                 # stack objects as tensor and aggregate other data for label
#                 boxes = torch.stack([torch.tensor([obj["x"],obj["y"],obj["l"],obj["w"],obj["h"],obj["direction"],obj["v"]]) for obj in [ts_data[key] for key in ts_data.keys()]])
#                 try:
#                     cam_ts_bias =  ts_data[list(ts_data.keys())[0]]["ts_bias"][camera.name]
#                 except KeyError:
#                     cam_ts_bias = 0
#                 # predict object positions assuming constant velocity
#                 dt = cam_ts + cam_ts_bias - ts
#                 boxes[:,0] += boxes[:,6] * dt * boxes[:,5] 
                
#                 # convert into image space
#                 im_boxes = self.hg.state_to_im(boxes,name = camera.name)
                
#                 # plot on frame
#                 camera.frame = self.hg.plot_state_boxes(camera.frame,boxes,name = camera.name,color = (255,0,0),secondary_color = (0,255,0),thickness = 2)
                
#                 #plot label
#                 classes = [ts_data[key]["class"] for key in ts_data.keys()]
#                 ids = [ts_data[key]["id"] for key in ts_data.keys()]
#                 speeds = [round(ts_data[key]["v"] * 3600/5280 * 10)/10 for key in ts_data.keys()]  # in mph
#                 directions = [ts_data[key]["direction"] for key in ts_data.keys()]
#                 directions = ["WB" if item == -1 else "EB" for item in directions]
#                 camera.frame = self.plot_labels(camera.frame,im_boxes,boxes,classes,ids,speeds,directions,ts+dt)
            
            
            
#             # concatenate frames
#             n_ims = len(cameras)
#             n_row = int(np.round(np.sqrt(n_ims)))
#             n_col = int(np.ceil(n_ims/n_row))
            
#             cat_im = np.zeros([1080*n_row,1920*n_col,3]).astype(float)
#             for im_idx, camera in enumerate(cameras):
#                 im = camera.frame
#                 row = im_idx // n_row
#                 col = im_idx % n_row
                
#                 cat_im[col*1080:(col+1)*1080,row*1920:(row+1)*1920,:] = im
            
            
#             # view frame and if necessary write to file
#             cat_im /= 255.0
#             #cat_im = cv2.resize(cat_im,(cat_im.shape[1]//2,cat_im.shape[0]//2))
#             cv2.imshow("frame",cat_im)
#             cv2.setWindowTitle("frame","At time{}".format(cam_ts))
#             key = cv2.waitKey(1)
#             if key == ord("p"):
#                 cv2.waitKey(0)
                
#             elif key == ord("q"):
#                 cv2.destroyAllWindows()
#                 for cap in cameras:
#                     cap.release()
#                     if savefile is not None:
#                         out.release()
#                 break
            
#             if savefile is not None:
#                 cat_im = cv2.resize(cat_im,(3840,2160))
#                 out.write(cat_im)
                
#             # advance one camera, rest will be advanced accordingly at beginning of loop
#             next(cameras[0])

#     def reinterpolate(self,frequency = 30,save = "reinterpolated_3D_tracking_outputs.csv"):
#         """
#         overwrites self copy of data with a regular sampling rate by interpolating object states
#         """            
        
#         ts_data,ts,next_ts,next_ts_data = next(self)
#         output_time = ts
#         start_time = ts
#         new_data = []
        
#         while next_ts is not None:
            
#             new_ts_data = {}
            
#             # for each object, interpolate position if present in both timestamps' data
#             for id in ts_data.keys():
#                 if id in next_ts_data.keys():
#                     obj = ts_data[id].copy()
#                     next_obj = next_ts_data[id]
                    
#                     # linear interpolation ratios
#                     r1 = (output_time - ts) / (next_ts - ts)
#                     r2 = 1 - r1
                    
#                     # interpolate changing fields
#                     for item in ["x","y","l","w","h","v"]:
#                         obj[item] = obj[item] * r1 + next_obj[item] * r2 
                    
#                     obj["timestamp"] = output_time
#                     new_ts_data[id] = obj
            
            
#             #append to data
#             new_data.append(new_ts_data)
        
#             output_time += 1.0/frequency
        
#             # if necessary, advance to next datum
#             while output_time > next_ts:
#                 ts_data,ts,next_ts,next_ts_data = next(self)
#                 if next_ts is None:
#                     break
#             if output_time < ts:
#                 print("Time Error!")
        
#         #overwrite data with new_data
#         self.data = new_data
#         self.d_idx = 0
        
#         if save is not None:
#             self.write_to_file(save_file = "reinterpolated_3D_tracking_outputs.csv")
            
#     def write_to_file(self,save_file = "default_save_file.csv"):
#         # create main data header
#         data_header = [
#             "Frame #",
#             "Timestamp",
#             "Object ID",
#             "Object class",
#             "BBox xmin",
#             "BBox ymin",
#             "BBox xmax",
#             "BBox ymax",
#             "vel_x",
#             "vel_y",
#             "Generation method",
#             "fbrx",
#             "fbry",
#             "fblx",
#             "fbly",
#             "bbrx",
#             "bbry",
#             "bblx",
#             "bbly",
#             "ftrx",
#             "ftry",
#             "ftlx",
#             "ftly",
#             "btrx",
#             "btry",
#             "btlx",
#             "btly",
#             "fbr_x",
#             "fbr_y",
#             "fbl_x",
#             "fbl_y",
#             "bbr_x",
#             "bbr_y",
#             "bbl_x",
#             "bbl_y",
#             "direction",
#             "camera",
#             "acceleration",
#             "speed",
#             "veh rear x",
#             "veh center y",
#             "theta",
#             "width",
#             "length",
#             "height",
#             "ts_bias for cameras {}".format(self.cameras)
#             ]

        
        
        
#         with open(save_file, mode='w') as f:
#             out = csv.writer(f, delimiter=',')
            
#             # write main chunk
#             out.writerow(data_header)
#             print("\n")
#             gen = "3D Detector"
#             camera = "p1c1" # default dummy value
            
#             for i,ts_data in enumerate(self.data):
#                 print("\rWriting outputs for time {} of {}".format(i,len(self.data), end = '\r', flush = True))

#                 for id in ts_data.keys():
#                     item = ts_data[id]
#                     id = item["id"]
#                     timestamp = item["timestamp"]
#                     cls = item["class"]
#                     ts_bias = item["ts_bias"]
#                     try:
#                         camera = item["camera"]
#                     except:
#                         camera = "p1c1"
#                     ts_bias = [ts_bias[key] for key in ts_bias.keys()]
#                     state = torch.tensor([item["x"],item["y"],item["l"],item["w"],item["h"],item["direction"],item["v"]])
                            
                        
#                     state = state.float()
                    
#                     if state[0] != 0:
                        
#                         # generate space coords
#                         space = self.hg.state_to_space(state.unsqueeze(0))
#                         space = space.squeeze(0)[:4,:2]
#                         flat_space = list(space.reshape(-1).data.numpy())
                        
#                         # generate im coords
#                         bbox_3D = self.hg.state_to_im(state.unsqueeze(0),name = camera)
#                         flat_3D = list(bbox_3D.squeeze(0).reshape(-1).data.numpy())
                        
#                         # generate im 2D bbox
#                         minx = torch.min(bbox_3D[:,:,0],dim = 1)[0].item()
#                         maxx = torch.max(bbox_3D[:,:,0],dim = 1)[0].item()
#                         miny = torch.min(bbox_3D[:,:,1],dim = 1)[0].item()
#                         maxy = torch.max(bbox_3D[:,:,1],dim = 1)[0].item()
                        
                        
#                         obj_line = []
                        
#                         obj_line.append("-") # frame number is not useful in this data
#                         obj_line.append(timestamp)
#                         obj_line.append(id)
#                         obj_line.append(cls)
#                         obj_line.append(minx)
#                         obj_line.append(miny)
#                         obj_line.append(maxx)
#                         obj_line.append(maxy)
#                         obj_line.append(0)
#                         obj_line.append(0)
    
#                         obj_line.append(gen)
#                         obj_line = obj_line + flat_3D + flat_space 
#                         state = state.data.numpy()
#                         obj_line.append(state[5])
                        
#                         obj_line.append(camera)
                        
#                         obj_line.append(0) # acceleration = 0 assumption
#                         obj_line.append(state[6])
#                         obj_line.append(state[0])
#                         obj_line.append(state[1])
#                         obj_line.append(np.pi/2.0 if state[5] == -1 else 0)
#                         obj_line.append(state[3])
#                         obj_line.append(state[2])
#                         obj_line.append(state[4])
    
    
#                         obj_line.append(ts_bias)
#                         out.writerow(obj_line)

# def test_integrity(sequence):
#     """
#     Counts number of doubled timestamps, doubled frames, and doubled-timestamps and frames
#     """
    
#     cam = Camera_Wrapper(sequence, ds = 1)
    
#     next(cam)
#     prev_ts = cam.ts
#     prev_frame = cam.frame.copy()
    
#     doubled_ts = 0
#     doubled_frame = 0
#     doubled_both = 0
#     correct = 0
#     skipped_ts = 0
#     n = 1000
    
    
#     for i in range(1,n):
        
#         next(cam)
#         DTS = False
#         DF = False
#         STS = False
#         ts = cam.ts
#         frame = cam.frame
        
#         if ts - prev_ts == 0:
#             DTS = True
        
#         if np.mean(np.abs(frame[100:500,100:500,:].astype(float) - prev_frame[100:500,100:500,:].astype(float))) < 0.2:
#             DF = True
            
#         if DTS and DF:
#             doubled_both += 1
#         elif DTS:
#             doubled_ts += 1
#         elif DF:
#             doubled_frame += 1
#         elif (ts - prev_ts) > 0.05:
#             skipped_ts += 1
#             STS = True
#         else:
#             correct += 1
            
#         if DTS or DF or STS:
#            #save both frames
#            cv2.imwrite("/home/worklab/Desktop/ex/{}_{}.png".format(cam.name,i),frame)
#            cv2.imwrite("/home/worklab/Desktop/ex/{}_{}.png".format(cam.name,i-1),prev_frame)
           
#            next(cam)
#            cv2.imwrite("/home/worklab/Desktop/ex/{}_{}.png".format(cam.name,i+1),cam.frame)
#            next(cam)
#            cv2.imwrite("/home/worklab/Desktop/ex/{}_{}.png".format(cam.name,i+2),cam.frame)


#         prev_frame = cam.frame.copy()
#         prev_ts = cam.ts
        
    
#         #print("On frame {} of {}, with {} errors so far".format(i,len(cam),(doubled_ts + doubled_frame + doubled_both)))
    
#     print("Camera {} results for {} frames:".format(cam.name,n))
#     print("Doubled timestamps occured {} times".format(doubled_ts))
#     print("Doubled both occured {} times".format(doubled_both))
#     print("Doubled frames occured {} times".format(doubled_frame))
#     print("Skipped timestamps occurred {} times".format(skipped_ts))

# if __name__ == "__main__":
           
#     data_csv = "/home/worklab/Documents/derek/3D-playground/_outputs/3D_tracking_results_10_27.csv"   
#     data_csv = "/home/worklab/Downloads/MC.csv"
#     #data_csv = "/home/worklab/Documents/derek/3D-playground/reinterpolated_3D_tracking_outputs.csv"        
#     #data_csv = "/home/worklab/Downloads/MC_rectified (1).csv"
    
#     sequences = ["/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c1_0_4k.mp4",
#                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c6_0_4k.mp4",
#                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c2_0_4k.mp4",
#                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c3_0_4k.mp4",
#                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c4_0_4k.mp4",
#                 "/home/worklab/Data/cv/video/ground_truth_video_06162021/segments_4k/p1c5_0_4k.mp4"]
    
#     # #with open("i24_all_homography.cpkl","rb") as f:
#     # #        hg = pickle.load(f)
#     # hg = Homography_Wrapper()
            
            
#     # dr = Data_Reader(data_csv,hg, metric = False)
#     # #dr.reinterpolate(save = None)
#     # dr.plot_in(sequences,framerate = 10)

#     for sequence in sequences:
#         test_integrity(sequence)