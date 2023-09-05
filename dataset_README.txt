Dataset file structure:

data/
   hg/
      scene1_hg.json
      scene2_hg.json
      scene3_hg.json
   mask/
      scene1/
         p1c1_mask.png       (4k)
         p1c1_mask_1080.png  (1080p)
         ...
      scene2/
         ...
      scene3/
         ...
   obj/
      scene1_annotations.csv
      scene2_annotations.csv
      scene3_annotations.csv
   
   spl_obj/
      scene1_splobj.json
      scene2_splobj.json
      scene3_splobj.json
   
   track/
      scene1_tracklets.json
      scene2_tracklets.json
      scene3_tracklets.json
   
   ts/
      scene1_ts.csv
      scene2_ts.csv
      scene3_ts.csv
      scene1_ts_orig.csv
      scene2_ts_orig.csv
      scene3_ts_orig.csv
      
   
video/
    scene1/
       p1c1.mp4
       p1c2.mp4
       ...
    scene2/
       ...
    scene3/
       ...
    
cache/
    This directory is empty but is designated as the default location for saving image files to disk
    
    
    
    
 
File Formats:

sceneX_hg.json

A dictionary of camera correspondences, keyed by camera name (e.g. "p1c1"). Each object is itself a dictionary with the following keys:
    corr_pts  - [n,2] array of correspondence points labeled in imagery
    space_pts - [n,2] array of correspondence points labeled in overhead roadway imagery (space). All points lie on z=0 plane by assumption.
    P         - [4,3] space-> image projection matrix
    H         - [3,3] image plane -> ground plane homography matrix
    H_inv     - [3,3] ground plane -> image plane homography matrix (3 of 4 columns are identical to P)
    curve     - 3 parameters fitting a second order polynomial to the curvature of the roadway, in imagery (see homography.py for usage)
    vps        - list of 3 pairs of points for (x,y) pixel coordinates for the longitudinal, lateral, and vertical vanishing points, respectively
    
    
sceneX_annotations.csv
Each file is sorted by increasing frame index. This is the required file format for scene.py 

frame	camera	id	x	y	l	w	h	direction	class	gen
0	p1c1	0	372.043	4.141	15.924	5.722	4.48	1		sedan	manual
0	p1c1	1	380.529	24.258	17.183	5.759	5.98	1		pickup	manual
0	p1c1	2	361.166	43.482	15.075	5.479	5.58	1		midsize	manual
0	p1c1	3	437.605	17.869	18.518	5.937	6.04	1		pickup	manual
0	p1c1	46	464.146	103.813	35.711	8.57	12.02	-1		truck	manual
0	p1c1	47	368.431	77.674	14.019	5.805	4.82	-1		midsize	manual
0	p1c1	48	321.267	90.856	14.507	6.365	4.86	-1		midsize	manual
...

direction - 1 for all EB (near side relative to cameras) vehicles and -1 for all WB vehicles
gen       - "manual" if point was manually clicked, "interpolation" if point was interpolated between manual points and subsequently inspected without modification, or "spline" if object was produced by sampling the spline approximation of the object
camera    - <pole#><camera#>
x,y,l,w,h - position and dimensions in feet
id        - unique integer for each unique object





sceneX_splobj.json and sceneX_tracklets.json

For tracking evaluation, objects are stored rather than storing data indexed by time. This allows for easy reinterpolation of the target and predicted object set to the same discrete times. This is the required format for evaluate.py.
Each file contains a list of objects. Each object is a dictionary with the following keys:

id 	   - int
class      - str   (from sedan, midsize, pickup, van, semi, truck (other))
l  	   - float (length in feet)
w  	   - float (width in feet)
h          - float (height in feet)
direction  - int (-1 or 1 for WB or EB)
x_position - 1D float array    (x position in feet)
y_position - 1D float array    (y position in feet)
timestamp  - 1D float array    (timestamp in s)






sceneX_ts.csv and sceneX_ts_orig.csv

First column is the frame index, all following columns are the corrected timestamp or original timestamp, respectively, in seconds, for each camera in the scene:
frame	p1c1	p1c2	p1c3	p1c4	p1c5	p1c6	p2c1	p2c2	p2c3	p2c4	p2c5	p2c6	p3c1	p3c2	p3c3	p3c4	p3c5
0	1623877088.8	1623877088.8075	1623877088.8213	1623877088.8193	1623877088.8284	1623877088.7236	1623877088.7908	1623877088.7235	1623877088.7934	1623877088.7335	1623877088.8369	1623877088.7406	1623877089.3395	1623877089.4021	1623877089.6324	1623877089.6127	1623877090.7898
1	1623877088.8357	1623877088.8404	1623877088.8527	1623877088.8492	1623877088.8685	1623877088.7636	1623877088.8309	1623877088.7635	1623877088.8305	1623877088.7677	1623877088.8727	1623877088.7734	1623877089.3651	1623877089.435	1623877089.6667	1623877089.6527	1623877090.8241
2	1623877088.8642	1623877088.8818	1623877088.8942	1623877088.8806	1623877088.9113	1623877088.798	1623877088.8738	1623877088.7949	1623877088.862	1623877088.8006	1623877088.9041	1623877088.8034	1623877089.3965	1623877089.4664	1623877089.6982	1623877089.6827	1623877090.8627
3	1623877088.8957	1623877088.9133	1623877088.9299	1623877088.9121	1623877088.9442	1623877088.8309	1623877088.9067	1623877088.8278	1623877088.8934	1623877088.8335	1623877088.9456	1623877088.8349	1623877089.4279	1623877089.4979	1623877089.7296	1623877089.7127	1623877090.8927
4	1623877088.9357	1623877088.9433	1623877088.9599	1623877088.9535	1623877088.9771	1623877088.8752	1623877088.9382	1623877088.8693	1623877088.9348	1623877088.8763	1623877088.9756	1623877088.8763	1623877089.4694	1623877089.5379	1623877089.7595	1623877089.7427	1623877090.9227
...
