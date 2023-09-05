# I24-3D-dataset
## Preliminaries
This repository provides basic example data usage utilities for the I-24 Multi-Camera 3D Dataset (I24-3D). The dataset is comprised of 3 scenes (sets of videos recorded at the same time from different cameras), recorded at 4K resolution and 30 frames per second across 16-17 cameras with dense viewpoints covering 2000 feet of interstate roadway near Nashville, TN. . Vehicle 3D bounding boxes are annotated by hand for 720 unique vehicles (over 877,000 3D vehicle bounding boxes annotated in total). I24-3D is the first 3D multiple-camera dataset in a traffic monitoring context and tracks objects continuously across a larger set of cameras than any other multi-camera tracking dataset (with possible exceptions for CityFlow, though this dataset contains gaps in camera coverage). 

This work is to appear at BMVC 2023. The preprint is available here: https://arxiv.org/pdf/2308.14833.pdf

[IMAGE HERE]

If you use this code or data, please cite this work and/or give the repo a star! Untold hours were put into this project and, while far from perfect, I am fairly happy with the pytorch-backended homography class (in `homography.py`) for managing many objects across many camera fields of view, and the openCV-backend video annotation framework (in `scene.py`), which is quite nicely extensible to various annotation regime after years of iteration on my part.

    @article{gloudemans2023i24dataset,
      title={The Interstate-24 3D Dataset: a new benchmark for 3D multi-camera vehicle tracking},
      author={Gloudemans, Derek and Wang, Yanbing and Gumm, Gracie and Barbour, William and Work, Daniel B},
      journal={arXiv preprint arXiv:2308.14833},
      year={2023}
    }



## Requirements
To use the dataset itself, there are no specific requirements (data is stored in .csv, .json, and .mp4 formats). To use the basic python scripts provided in this repository, you'll need a few python libraries:
- `pytorch`
- `numpy`
- `cv2`



## TODO
- [X] Save annotations as flat file
- [X] Save homography data as flat file
- [X] Save spline object data as flat file
- [X] Save timestamps as flat file
- [X] Nest homgraphy code nicely
- [X] Update homography to use curve offset parameters and load from JSON
- [X] Implement CSV - dataloader
- [X] Implement JSON spline dataloader
- [X] Implement timestamp file dataloader
- [X] Make a simple DataHandler object that serves up annotations, timestamps and frames with sweet sweet precision and does basic things
- [X] Make a simple viz tool that shows objects, spline objects, masks, roadway grid, and object details
- [X] Make a no-handrails annotation tool

- [ ] Cache evaluation runs from database as files
- [ ] Get evaluation script to work with the above file structure
- [ ] Stash all previous code and data on a hard drive never to be seen or used again!
- [ ] Make a beautiful beautiful readme
- [ ] Update the manuscript


## Notes
- `cv2`-based video decoding is painfully slow (decoding 16ish 4k video buffers on cpu generally proceeds at less than 1 fps). Moreover, cv2 `VideoCapture` objects don't provide an easy way to jump to a point in the video, so you're stuck retrieving each frame to get to later frames. For faster decoding (~10 fps) you can write an asynchronous loader or try out [Nvidia's VPF library](https://github.com/NVIDIA/VideoProcessingFramework) for GPU side decoding directly into pytorch tensors. This library isn't utilized in this work because the installation can be quite painful and has soft-failure dependencies.
- Buffered 4K frames quickly fill memory, making this dataset a bit of a pain to work with in video form. 
- For speed and buffer memory reasons, it is recommended to save frames to disk using the provided script whenever working with data from a single frame or batches of single frames at a time.
- Ground truth data of this size is likely to contain some annotation errors. Do not raise these as issues as they will quickly overwhelm the feed and be lost. If you care to report or correct these issues to improve data quality, please do so by adding a line to the `issue_tracker.csv` reporting the issue and submit a pull request.


