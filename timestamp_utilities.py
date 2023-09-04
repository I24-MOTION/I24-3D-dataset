import os
import re
import pickle
import cv2
import ast
from numpy import array
#from parameters import *
    

def get_precomputed_checksums(abs_path=None):
    path = './resources/timestamp_pixel_checksum_6.pkl'
    if abs_path is not None:
        path = abs_path
        
    with open(path, 'rb') as pf:
        dig_cs6 = pickle.load(pf)
        
    return dig_cs6


def get_timestamp_geometry(abs_path=None):
    path = './resources/timestamp_geometry_4K.pkl'
    if abs_path is not None:
        path = abs_path
        
    with open(path, 'rb') as pf:
        g = pickle.load(pf)
    return g


def get_timestamp_pixel_limits():
    """
    Provides x/y coordinates (only) for timestamp pixel extraction. Note that return order is y1, y2, x1, x2. Timestamp
        can be extracted from full frame like so: `timestamp_pixels = frame_pixels[y1:y2, x1:x2, :]`
    :return: y-start (y1), y-stop (y2), x-start (x1), x-stop (x2)
    """
    geom = get_timestamp_geometry()
    y0 = geom['y0']
    x0 = geom['x0']
    h = geom['h']
    n = geom['n']
    w = geom['w']
    return y0, y0+h, x0, x0+(n*w)


def parse_frame_timestamp(timestamp_geometry, precomputed_checksums, frame_pixels=None, timestamp_pixels=None):
    """
    Use pixel checksum method to parse timestamp from video frame. First extracts timestamp area from frame
        array. Then converts to gray-scale, then converts to binary (black/white) mask. Each digit
        (monospaced) is then compared against the pre-computed pixel checksum values for an exact match.
    :param timestamp_geometry: dictionary of parameters used for determining area of each digit in checksum
        (load using utilities.get_timestamp_geometry)
    :param precomputed_checksums: dictionary of checksum:digit pairs (load using utilities.get_precomputed_checksums())
    :param frame_pixels: numpy array of full (4K) color video frame; dimensions should be 2160x3840x3
    :param timestamp_pixels: numpy array of timestamp area, defined by `get_timestamp_pixel_limits()`
    :return: timestamp (None if checksum error), pixels from error digit (if no exact checksum match)
    """
    # extract geometry values from timestamp_geometry dictionary
    g = timestamp_geometry
    w = g['w']
    h = g['h']
    x0 = g['x0']
    y0 = g['y0']
    n = g['n']
    h13 = g['h13']
    h23 = g['h23']
    h12 = g['h12']
    w12 = g['w12']
        
    dig_cs6 = precomputed_checksums
    
    if frame_pixels is not None:
        # extract the timestamp in the x/y directions
        tsimg = frame_pixels[y0:(y0+h), x0:(x0+(n*w)), :]
    elif timestamp_pixels is not None:
        tsimg = timestamp_pixels
    else:
        raise ValueError("One of `frame_pixels` or `timestamp_pixels` must be specified.")
    # convert color to gray-scale
    tsgray = cv2.cvtColor(tsimg, cv2.COLOR_BGR2GRAY)
    # convert to black/white binary mask using fixed threshold (1/2 intensity)
    # observed gray values on the edges of some digits in some frames were well below this threshold
    ret, tsmask = cv2.threshold(tsgray, 127, 255, cv2.THRESH_BINARY)

    # parse each of the `n` digits
    ts_dig = []
    for j in range(n):
        # disregard the decimal point in the UNIX time (always reported in .00 precision)
        if j == 10:
            ts_dig.append('.')
            continue

        # extract the digit for this index, was already cropped x0:x0+n*w, y0:y0+h
        pixels = tsmask[:, j*w:(j+1)*w]
        
        # cv2.imshow("pixels",pixels)
        # cv2.waitKey(0)
        
        # compute the 6-area checksum and convert it to an array
        cs = [[int(pixels[:h13, :w12].sum() / 255), int(pixels[:h13, w12:].sum() / 255)],
              [int(pixels[h13:h23, :w12].sum() / 255), int(pixels[h13:h23, w12:].sum() / 255)],
              [int(pixels[h23:, :w12].sum() / 255), int(pixels[h23:, w12:].sum() / 255)]
              ]
        cs = array(cs)
        # compute the absolute difference between this digit and each candidate
        cs_diff = [(dig, abs(cs - cs_ref).sum()) for dig, cs_ref in dig_cs6.items()]
        pred_dig, pred_err = min(cs_diff, key=lambda x: x[1])
        # looking for a perfect checksum match; testing showed this was reliable
        if pred_err > 0:
            # if no exact match, return no timestamp and the pixel values that resulted in the error
            return None, pixels
        else:
            ts_dig.append(pred_dig)
    # convert the list of strings into a number and return successful timestamp
    return ast.literal_eval(''.join(map(str, ts_dig))), None


def parse_config_file(config_file):
    """
    Parses an entire session configuration file into sections (in this order): cameras, image snapshot, video snapshot,
        and recording.
    :param config_file: path to configuration file
    :return: configuration dictionaries of key-value pairs; list of dicts for cameras section, single dict for others.
    """
    camera_config = []
    image_snap_config = []
    video_snap_config = []
    recording_config = []
    block_mapping = {'__CAMERA__': camera_config,
                     '__IMAGE-SNAPSHOT__': image_snap_config,
                     '__VIDEO-SNAPSHOT__': video_snap_config,
                     '__PERSISTENT-RECORDING__': recording_config}
    # open configuration file and parse it out
    with open(config_file, 'r') as f:
        current_block = None
        block_destination = None
        for line in f:
            # ignore empty lines and comment lines
            if line is None or len(line.strip()) == 0 or line[0] == '#':
                continue
            strip_line = line.strip()
            if len(strip_line) > 2 and strip_line[:2] == '__' and strip_line[-2:] == '__':
                # this is a configuration block line
                # first check if this is the first one or not
                if block_destination is not None and len(current_block) > 0:
                    # add the block to its destination if it's non-empty
                    block_destination.append(current_block)
                # reset current block to empty and set its destination
                current_block = {}
                block_destination = block_mapping[strip_line]
            elif '==' in strip_line:
                pkey, pval = strip_line.split('==')
                current_block[pkey.strip()] = pval.strip()
            else:
                raise AttributeError("""Got a line in the configuration file that isn't a block header nor a 
                key=value.\nLine: {}""".format(strip_line))
        # add the last block of the file (if it's non-empty)
        if block_destination is not None and len(current_block) > 0:
            block_destination.append(current_block)

    # check number of configuration blocks for these configs
    if len(image_snap_config) > 1:
        raise AttributeError("More than one configuration block found for __IMAGE-SNAPSHOT__.")
    elif len(image_snap_config) == 1:     # had one config block
        image_snap_config = image_snap_config[0]
    if len(video_snap_config) > 1:
        raise AttributeError("More than one configuration block found for __VIDEO-SNAPSHOT__.")
    elif len(video_snap_config) == 1:     # had one config block
        video_snap_config = video_snap_config[0]
    if len(recording_config) > 1:
        raise AttributeError("More than one configuration block found for __PERSISTENT-RECORDING__.")
    elif len(recording_config) == 1:     # had one config block
        recording_config = recording_config[0]

    # send back configs
    return camera_config, image_snap_config, video_snap_config, recording_config


def get_session_start_time_local(session_info_filename):
    """
    Finds the local time at which the session was started, according to the _SESSION_INFO.txt file.
    :param session_info_filename: path to session info filename
    :return: datetime.datetime object representing local time at which session was initialized
    """
    import datetime
    with open(session_info_filename, 'r') as f:
        for line in f:
            if line.startswith("Session initialization time (local): "):
                ts = line.strip("Session initialization time (local): ")
                ts = datetime.datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S.%f")
                break
        else:
            raise ValueError("Couldn't find line with timestamp.")
    return ts


def get_sesssion_recording_segment_time(session_info_filename):
    """
    Finds the duration of video segments for perpetual recording during this session. This info is in the _SESSION_INFO
        file, as well as the _SESSION_CONFIG file. Pulled from INFO file for convenience in this case.
    :param session_info_filename:  path to session info filename
    :return: number of minutes for video segments
    """
    with open(session_info_filename, 'r') as f:
        for line in f:
            if line.startswith("Recording segment duration: "):
                rst = float(line.strip("Recording segment duration: "))
                break
        else:
            raise ValueError("Couldn't find line with recording duration.")
    return rst


def get_session_number(session_info_filename=None):
    """
    Finds the session number, according to the _SESSION_INFO.txt file.
    :param session_info_filename: path to session info filename
    :return: session number (integer)
    """
    with open(session_info_filename, 'r') as f:
        for line in f:
            if line.startswith("SESSION #"):
                sn = line.strip("SESSION #")
                break
        else:
            raise ValueError("Couldn't find line with session number.")
    return int(sn)


def get_recording_params(session_root_directory, session_number=None, camera_configs=None, recording_config=None,verbose = True):
    """
    Determine relevant parameters from video ingest session configuration: list of recording directories where video
        files are stored (corresponding to each camera, might be the same), file name formatter (corresponding to each
        camera, might be the same), list of camera names. Note: length of all return lists = number of cameras.
        Providing configuration dictionaries as inputs is optional; if either is left as None, the the session
        configuration will be loaded and parsed automatically from _SESSION_CONFIG.config. Providing session_number is
        also optional; it will be loaded from _SESSION_INFO.txt if not provided.
    :param session_root_directory: directory of video ingest session, which contains automatic copy of config file
    :param session_number: (optional) session number corresponding to this directory
    :param camera_configs: (optional) list of camera configuration dictionaries (used to get camera names)
    :param recording_config: (optional) recording configuration dictionary (used to get recording file name format)
    :param verbose: bool - allow or supress function print statements
    :return: list of recording directories for each camera, file name format for each camera, list of camera names
    """
    if camera_configs is None or recording_config is None:
        if verbose: print("Loading configuration file instead of using configuration input arguments.")
        camera_configs, _, _, recording_config = parse_config_file(
            config_file=os.path.join(session_root_directory, "_SESSION_CONFIG.config"))
    if session_number is None:
        session_number = get_session_number(
            session_info_filename=os.path.join(session_root_directory, DEFAULT_SESSION_INFO_FILENAME))
    # get camera names for filename formatting
    cam_names = []
    for single_camera_config in camera_configs:
        cam_names.append(single_camera_config['name'])
    #print("All camera names: {}".format(cam_names))
    
    # get the recording filename, or the default
    file_location = recording_config.get('recording_filename', DEFAULT_RECORDING_FILENAME)
    # split path location into directory and filename
    file_dir, file_name = os.path.split(file_location)
    #("File template name: {}".format(file_name))
    #print("File location: {}".format(file_location))
    
    # check if it's a relative file path, and if so change it to absolute using session_root_directory
    if file_dir.startswith('./'):
        file_dir = os.path.join(session_root_directory, file_dir[2:])
    # put session number and camera name in directory or filename, if indicated
    rec_dirs = [file_dir.format(cam_name=cam_name, session_num=session_number) for cam_name in cam_names]
    file_names = [file_name.format(cam_name=cam_name, session_num=session_number) for cam_name in cam_names]
    # list of recording directories corresponding to each camera (might all be the same if not delineated by camera)
    # list of file names corresponding to each camera (might all be the same if not delineated by camera)
    # list of camera names
    return rec_dirs, file_names, cam_names


def find_files(recording_directories, file_name_formats, camera_names, drop_last_file=False, first_file_index=0,
               filter_filenames=None, verbose = True):
    """
    Determine files in recording directories that match file recording naming format. This function is written to be
        used immediately following get_recording_params(...). The output of that function can be used as the inputs
        `recording_directories`, `file_name_formats`, `camera_names` to this function. The directories and file name
        formats should correspond in length and order to the camera_names, though one of those lists may contain all
        the same items (e.g., './recording' for the recording directories).
    :param recording_directories: list of directories in which to search for files; each corresponding to camera_names.
    :param file_name_formats: list of file name formats used for recording; each corresponding to camera_names.
    :param camera_names: list of camera names to substitute into file name format
    :param drop_last_file: flag to ignore/drop the last file in the recording sequence, per camera
    :param first_file_index: minimum recording segment number to keep files (used for checking recent files only)
    :param filter_filenames: list of filters to narrow down filenames (tested by `if any filter in filename`)
    :param verbose: bool - allow or supress function print statements
    :return: list of tuples of form (file directory, filename, segment_number, camera_name) for matching recordings
    """
    file_name_regexs = [re.sub('%(0[0-9]{1})*d', '([0-9]+)', fnf) for fnf in file_name_formats]
    match_files = []
    for cn, rdir, fnr in zip(camera_names, recording_directories, file_name_regexs):
        directory_files = os.listdir(rdir)
        if verbose: print("Searching {} files in directory {} to match {} (camera {}).".format(len(directory_files), rdir, fnr, cn))
        cam_files = []
        for fl in directory_files:
            rem = re.search(fnr, fl)
            if rem is not None:
                # extract the first group match, which contains the segment index
                remi = int(rem.group(1))
                if remi >= first_file_index:
                    cam_files.append((rdir, fl, remi, cn))
        # sort files by segment index and drop the last one, if requested, while adding to all matches
        if drop_last_file is True:
            match_files += sorted(cam_files, key=lambda x: x[2])[:-1]
            if verbose: print("Found {} matching files.".format(len(cam_files) - 1))
        else:
            match_files += sorted(cam_files, key=lambda x: x[2])
            if verbose: print("Found {} matching files.".format(len(cam_files)))
    if filter_filenames is not None:
        match_files = [fn for fn in match_files if
                       any([fn_filt in os.path.join(fn[0], fn[1]) for fn_filt in filter_filenames])]
        if verbose: print("Filtered files to {} matching.".format(len(match_files)))
    return match_files


def get_manager_log_files(session_directory, log_directory=None):
    """
    Determines list of log files written by video ingest manager.
    :param session_directory: top level directory of video ingest session
    :param log_directory: location of log files from video ingest manager (overrides session_directory)
    :return: list of files matching ingest manager logs ("manager-TIMESTAMP.log)
    """
    look_in_directory = (os.path.join(session_directory, 'logs') if log_directory is None else log_directory)
    manager_logs = []
    for fn in os.listdir(look_in_directory):
        if re.search('manager-(.*)\.log', fn) is not None:
            manager_logs.append(fn)
    return manager_logs
