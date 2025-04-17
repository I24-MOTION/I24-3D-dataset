import cv2
import time
import numpy as np
from torchvision.ops import nms 
import torch
import os

import imp
MainModel = imp.load_source('MainModel', "load_model.py")

from scene import Scene


class blacker():
    def __init__(self, read_path, write_path):
        self.vid = cv2.VideoCapture(read_path)
        
        self.crop_size = np.array([300, 300])
        self.writer = cv2.VideoWriter(write_path, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'), self.vid.get(cv2.CAP_PROP_FPS), (3840,2160))
        self.model = torch.load('plate.pth').to('cuda')
        self.count = 0
    
    # Black out the license plates in the input image COMPLETE
    def black(self, crop, pts):
        pts = pts.data.numpy().astype(int)
        crop = cv2.fillPoly(crop, [pts], (0, 0, 0))


    def decode(self, result, threshold=0.1):
        result = result.to('cpu').numpy()
        net_stride = 2**4
        side = 7.75

        Probs = result[..., 0]
        Affines = result[...,-6:]

        try:
            xx, yy = np.where(Probs>threshold)
            #print('hey')
            #print(xx)
            if(len(xx) == 0):
                #print(Probs)
                #print(max(Probs))
                return False
        except:
            return False

        WH = self.crop_size
        MN = WH/net_stride

        base = np.matrix([[-.5,-.5,1.],[.5,-.5,1.],[.5,.5,1.],[-.5,.5,1.]]).T
        conf = np.zeros((len(xx)))
        ptss = np.zeros((len(xx), 4, 2))

        for i in range(len(xx)):
            y, x = xx[i], yy[i]
            affine = Affines[y, x]
            prob = Probs[y, x]

            mn = np.array([float(x) + .5, float(y) + .5])

            A = np.reshape(affine, (2, 3))
            A[0, 0] = max(0., A[0, 0])
            A[1, 1] = max(0., A[1, 1])
            pts = np.array(A*base)
            pts_MN_center_mn = pts * side
            pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
            
            pts_prop = pts_MN/MN.reshape((2, 1))

            pts_prop = pts_prop.T.reshape(4, 2)

            conf[i] = prob
            ptss[i] = pts_prop

        return ptss[np.argmax(conf)]
        

    def crop_vehicle(self, frame, bboxes):
        crops = []
        for box in bboxes:
            crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2]), :]
            if 0 in crop.shape:
                continue
            crops.append(crop)
        return crops

    def plate_detection(self, crops):
        
        if len(crops) == 0:
            return [], []
        
        crops_resized = torch.zeros((len(crops), 3, self.crop_size[0], self.crop_size[1]), device='cuda')
        for i in range(len(crops)):
            h, w = crops[i].shape[:2]
            resize_dim = []
            pad_dim = []
            if h > w:
                resize_dim = (int(self.crop_size[1] * w / h), self.crop_size[0])
                pad_dim =  (0, self.crop_size[1] - resize_dim[0] , 0, 0)
            else:
                resize_dim = (self.crop_size[1], int(self.crop_size[0] * h / w))
                pad_dim = (0, 0, 0, self.crop_size[0] - resize_dim[1])

            resized = torch.from_numpy(cv2.resize(crops[i], resize_dim)).permute(2, 0, 1).to('cuda')
            resized = torch.nn.functional.pad(resized, pad_dim).type(torch.float32) / 255
            crops_resized[i] = resized

        with torch.no_grad():
            results = self.model(crops_resized)

        results = results.permute(0, 2, 3, 1)
        out = []
        out_idx = []
        for i in range(len(crops)):
            pts = self.decode(results[i], 0.1)
            if isinstance(pts, bool):
                continue
            long_side = max(crops[i].shape[:2])
            pts = (pts * long_side).astype(np.int32)
            out.append(pts)
            out_idx.append(i)
            
        return out,out_idx

    def step(self,bboxes):
        
        # advance frame
        ret, frame = self.vid.read()
        
        if len(bboxes) > 0:
            
            # get crops
            crops = self.crop_vehicle(frame, bboxes)
            
            regions,idxs = self.plate_detection(crops)
            
            
            # local -> global transformation
            if len(regions) > 0:
                global_regions = self.local_to_global(regions,bboxes[idxs,:])
                # redact
                for region in global_regions:
                    self.black(frame,region)
                
        
        self.writer.write(frame)
    
    def local_to_global(self,preds,bboxes):
        """
        Convert from crop coordinates to frame coordinates
        preds - [n,d,20] array where n indexes object and d indexes detections for that object
        crops_boxes - [n,4] array
        """
        
        if len(preds) == 0:
            return
        else:
            preds = torch.from_numpy(np.stack(preds))
        n = preds.shape[0]
        preds = preds.reshape(n,4,2)
        
        scales = torch.max(torch.stack([bboxes[:,2] - bboxes[:,0],bboxes[:,3] - bboxes[:,1]]),dim = 0)[0]
        scales = scales.unsqueeze(1).unsqueeze(1).expand(n,4,2)
        
        # scale each box by the box scale / crop size self.cs
        preds = preds * scales / self.crop_size
    
        # shift based on crop box corner
        preds[:,:,0] += bboxes[:,0].unsqueeze(1).expand(n,4)
        preds[:,:,1] += bboxes[:,1].unsqueeze(1).expand(n,4)
        
        return preds
        

if __name__ == '__main__':
    
    for scene_id in [1,2,3]:
        scene_length = 2700 if scene_id == 1 else 1800
        
        video_dir = "/home/worklab/Documents/I24-3D/video"
        data_dir  = "/home/worklab/Documents/I24-3D/data"
        scene = Scene(video_dir,data_dir,scene_id = scene_id)

        video_dir = "/home/worklab/Documents/I24-3D/video/scene{}".format(scene_id)
        out_dir   = "/home/worklab/Documents/I24-3D/redacted/scene{}".format(scene_id)
        
        vids = os.listdir(video_dir)
        vids.sort()
        for vid in vids:
            out_vid = os.path.join(out_dir,vid)
            vid = os.path.join(video_dir,vid)
            
            cam = vid.split("/")[-1].split(".")[0]

            if os.path.exists(out_vid): continue
        
    
            black = blacker(vid,out_vid)
            for f_idx in range(scene_length):
                if f_idx % 100 == 0:
                    print("On sequence {} frame {}".format(vid,f_idx))
                # get objects
                objs = scene.data[f_idx].values()
                objs = list(filter(lambda obj: obj["camera"] == cam,objs))
                
        
                if len(objs) > 0:
                    boxes = torch.stack([torch.tensor([obj["x"], obj["y"], obj["l"], obj["w"], obj["h"], obj["direction"]]).float() for obj in objs])
                
                    im_boxes = scene.hg.state_to_im(boxes,name = cam) * 2
                    #fbr,fbl,bbr,bbl,ftr,ftl,fbr,fbl
                    # crop the back face of the 3D bounding box for each vehicle
                    minx,_  = torch.min(im_boxes[:,[2,3,6,7],0],dim = 1)
                    maxx,_  = torch.max(im_boxes[:,[2,3,6,7],0],dim = 1)
                    miny,_  = torch.min(im_boxes[:,[2,3,6,7],1],dim = 1)
                    maxy,_  = torch.max(im_boxes[:,[2,3,6,7],1],dim = 1)
                    
                    im_boxes = torch.stack([minx,miny,maxx,maxy]).transpose(1,0)
                    # be sure to check scale 
                    
                else:
                    im_boxes = []
               
                black.step(im_boxes)
                

                
                if f_idx > 100:
                    break
                
            black.writer.release()
            black.vid.release() 