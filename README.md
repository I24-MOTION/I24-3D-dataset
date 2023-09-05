# I24-3D-dataset

This work is to appear at BMVC 2023.
paper: https://arxiv.org/pdf/2308.14833.pdf

## Preliminaries


## Requirements
To use just the dataset itself, there are no specific requirements. To use the basic python scripts provided in this repository, you'll need a few python libraries:
- `pytorch`
- `numpy`
- `cv2`


## Notes
- `cv2`-based video decoding is painfully slow (decoding 16ish 4k video buffers on cpu generally proceeds at less than 1 fps). Moreover, cv2 `VideoCapture` objects don't provide an easy way to jump to a point in the video, so you're stuck retrieving each frame to get to later frames. For faster decoding (~10 fps) you can write an asynchronous loader or try out [Nvidia's VPF library](https://github.com/NVIDIA/VideoProcessingFramework) for GPU side decoding directly into pytorch tensors. This library isn't utilized in this work because the installation can be quite painful and has soft-failure dependencies.
- Buffered 4K frames quickly fill memory, making this dataset a bit of a pain to work with in video form. 
- For speed and buffer memory reasons, it is recommended to save frames to disk using the provided script whenever working with data from a single frame or batches of single frames at a time.
  

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
