import cv2
from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Optional, Dict
from multiprocessing import Queue, Process, Event 
from abc import ABC
from tqdm import tqdm
from image_tools import im2single, im2gray
# TODO: add resizing as an option 
# TODO: Check index error (+-1). Make sure that number of frames is correct (end index valid)

class VideoReader(ABC):

    def open_file(            
            filename: str, 
            safe: bool = False
        ) -> None:
        pass

    def next_frame(self) -> Tuple[bool,NDArray]:
        pass
        
    def get_fps(self) -> float:
        pass
    
    def get_width(self) -> int:
        pass
    
    def get_height(self) -> int:
        pass
    
    def get_num_channels(self) -> int:
        pass

    def get_number_of_frame(self) -> int:
        pass

    def close(self) -> None:
        pass

def get_number_of_frames(filename: str, safe: bool = True):

    cap = cv2.VideoCapture(filename)

    # number of frames
    if safe:
        # loop over the whole video and count each frame
        counter = 0
        while True:
            rval, frame = cap.read()
            if not rval:
                break
            counter = counter + 1
        number_of_frames = counter

    else:
        # Trust opencv to return video properties. This is fast but there are known issues 
        # with this. Use at your own risk.
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return number_of_frames 

def get_fps(filename: str) -> int:
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_image_properties(filename: str, safe: bool = True) -> Tuple[int, int, int, np.dtype]:

    cap = cv2.VideoCapture(filename)
    rval, frame = cap.read()
    if not rval:
        raise RuntimeError('Could not read first frame')

    height, width, num_channels = frame.shape
    data_type = frame.dtype

    cap.release()
    return (height, width, num_channels, data_type)

def get_video_info(filename: str, safe: bool = True) -> Dict:
    num_frames = get_number_of_frames(filename, safe)
    fps = get_fps(filename)
    height, width, num_channels, data_type = get_image_properties(filename, safe)
    info = {
        'height': height,
        'width': width,
        'num_channels': num_channels,
        'data_type': data_type,
        'fps': fps,
        'num_frames': num_frames,
    }
    return info

class OpenCV_VideoReader(VideoReader):
    
    def __init__(self):
        self.open = False

    def open_file(
            self, 
            filename: str, 
            safe: bool = False, 
            resize: float = 1,
            crop: Optional[Tuple[int,int,int,int]] = None,
            backend: Optional[int] = None
        ) -> None:
            
        # store param
        self._filename = filename
        self._safe = safe
        self._current_frame = 0
        self._crop = crop # [left,bottom,width,height]
        self._resize = resize
        self._backend = backend

        # get video metadata 
        self._video_info = get_video_info(filename, safe)
        self._number_of_frames = self._video_info['num_frames']
        self._fps = self._video_info['fps']
        self._num_channels = self._video_info['num_channels']
        self._type = self._video_info['data_type']
        if crop is not None:
            self._width = int(crop[2] * resize)
            self._height = int(crop[3] * resize)
        else:
            self._width = self._video_info['width'] * resize
            self._height = self._video_info['height'] * resize
            
        if backend is not None:
            self._capture = cv2.VideoCapture(filename, backend)
        else:
            self._capture = cv2.VideoCapture(filename)
        self.open = True

    def close(self):
        self._capture.release()
        self.open = False

    def is_open(self) -> bool:
        return self.open
    
    def reset_reader(self) -> None:
        """
        TODO
        """
        self._capture.release()
        if self._backend is not None:
            self._capture = cv2.VideoCapture(self._filename, self._backend)
        else:
            self._capture = cv2.VideoCapture(self._filename)
        self._current_frame = 0

    def next_frame(self) -> Tuple[bool,NDArray]:
        """
        TODO
        """
        rval, frame = self._capture.read()
        if rval:
            self._current_frame = self._current_frame + 1
            if self._crop is not None:
                frame = frame[
                    self._crop[1]:self._crop[1]+self._crop[3],
                    self._crop[0]:self._crop[0]+self._crop[2]
                ]
            if self._resize != 1:
                frame = cv2.resize(
                    frame,
                    None,
                    None,
                    self._resize,
                    self._resize,
                    cv2.INTER_NEAREST
                )
        return (rval, frame)
    
    def previous_frame(self) -> Tuple[bool,NDArray]:
        self.seek_to(self._current_frame - 1) # -1 or -2 ?
        return self.next_frame()
    
    def seek_to(self, index):

        # check arguments value
        if not 0 <= index < self._number_of_frames:
            raise(ValueError(f"index should be between 0 and {self._number_of_frames-1}, got {index}"))
        
        if self._safe:
            # if you need to rewind, start from the beginning, otherwise keep going
            if index < self._current_frame:
                # reinitialize video reader
                self.reset_reader()

            # go through the video until index
            counter = self._current_frame
            while counter < index-1:
                rval, frame = self.next_frame()
                if not rval:
                    raise(RuntimeError(f"movie ended while seekeing to frame {index}"))
                counter = counter + 1
 
        else:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index-1)
            self._current_frame = index-1
    
    def read_frames(self, start: int, stop: int) -> NDArray:
        """
        Read frames between indices start and stop and store them in a numpy array in RAM
        """

        # check arguments value
        if not 0 <= start < self._number_of_frames:
            raise(ValueError(f"start index should be between 0 and {self._number_of_frames-1}"))
        if not start <= stop < self._number_of_frames:
            raise(ValueError(f"stop index should be between {start} and {self._number_of_frames-1}"))
        
        self.seek_to(start)

        # preinitialize arrray. Be aware that this could take a lot of RAM depending on 
        # resolution and number of frames
        if self._num_channels > 1:
            frames = np.empty((self._height, self._width, self._num_channels, stop-start), dtype=self._type)
        else:
            frames = np.empty((self._height, self._width, stop-start), dtype=self._type)

        # read frames
        counter = self._current_frame
        while counter < stop:
            rval, frame = self.next_frame()
            if not rval:
                raise(RuntimeError(f"movie ended while seeking to frame {stop}"))
            frames[:,:,:,counter-start] = frame
            counter = counter + 1

        return frames

    def get_width_max(self) -> int:
        return self._video_info['width']

    def get_height_max(self) -> int:
        return self._video_info['height']
    
    def get_fps(self) -> float:
        return self._fps
    
    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height
    
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_type(self) -> np.dtype:
        return self._type
    
    def get_current_frame_index(self) -> int:
        return self._current_frame

    def get_filename(self) -> str:
        return self._filename
    
    def get_number_of_frame(self) -> int:
        return self._number_of_frames

class Buffered_OpenCV_VideoReader(Process, VideoReader):
    '''
    Buffer video file reading. Cannot seek to a specific frame. Faster 
    when you need to process each video frame sequentially
    '''

    def open_file(
            self, 
            filename: str, 
            safe: bool = False, 
            resize: float = 1,
            crop: Optional[Tuple[int,int,int,int]] = None,
            backend: Optional[int] = None
        ):
            
        # store param
        self._filename = filename
        self._safe = safe
        self._current_frame = 0
        self._crop = crop # [left,bottom,width,height]
        self._resize = resize
        self._backend = backend

        # get video metadata 
        self._video_info = get_video_info(filename, safe)
        self._number_of_frames = self._video_info['num_frames']
        self._fps = self._video_info['fps']
        self._num_channels = self._video_info['num_channels']
        self._type = self._video_info['data_type']
        if crop is not None:
            self._width = int(crop[2] * resize)
            self._height = int(crop[3] * resize)
        else:
            self._width = self._video_info['width'] * resize
            self._height = self._video_info['height'] * resize

        # multiprocessing
        self._queue = Queue(maxsize=100) 
        self._stop = Event()
        self.start()

    def close(self):
        self._stop.set()

    def run(self) -> None:
        if self._backend is not None:
            self._capture = cv2.VideoCapture(self._filename)
        else:
            self._capture = cv2.VideoCapture(self._filename, self._backend)
        self._current_frame = 0
        while not self._stop.is_set():
            rval, frame = self.read_frame()
            if not rval:
                break
            self._queue.put((rval, frame))
        self._capture.release()
            
    def next_frame(self):
        return self._queue.get()

    def read_frame(self) -> Tuple[bool,NDArray]:

        rval, frame = self._capture.read()
        if rval:
            self._current_frame = self._current_frame + 1
            if self._crop is not None:
                frame = frame[
                    self._crop[1]:self._crop[1]+self._crop[3],
                    self._crop[0]:self._crop[0]+self._crop[2]
                ]
            if self._resize != 1:
                frame = cv2.resize(
                    frame,
                    None,
                    None,
                    self._resize,
                    self._resize,
                    cv2.INTER_NEAREST
                )
        return (rval, frame)

    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height

    def get_fps(self) -> float:
        return self._fps
        
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_type(self) -> np.dtype:
        return self._type
    
    def get_current_frame_index(self) -> int:
        return self._current_frame

    def get_filename(self) -> str:
        return self._filename
    
    def get_number_of_frame(self) -> int:
        return self._number_of_frames
    
class InMemory_OpenCV_VideoReader(VideoReader):
    '''
    Decode video file and buffer raw frames in memory.
    This is useful when you want to test the performance 
    of a code that acts on frames but don't want to be limited 
    by how fast you can read your video file.
    Please be aware that for large pixel counts, raw frames take
    up a lot of space.
    Specify memsize_bytes to limit memory used
    '''
    
    def __init__(self):
        self.open = False

    def open_file(
            self, 
            filename: str, 
            safe: bool = False, 
            memsize_bytes: int = 1e9, # one GB by default
            resize: float = 1,
            crop: Optional[Tuple[int,int,int,int]] = None,
            backend: Optional[int] = None,
            single_precision: bool = False,
            grayscale: bool = False
        ) -> None:
            
        # store param
        self._filename = filename
        self._safe = safe
        self._memsize_bytes = memsize_bytes
        self._current_frame = 0
        self._crop = crop # [left,bottom,width,height]
        self._resize = resize
        self._backend = backend
        self._single_precision = single_precision
        self._grayscale = grayscale

        # get video metadata 
        self._video_info = get_video_info(filename, safe)
        self._number_of_frames = self._video_info['num_frames']
        self._fps = self._video_info['fps']

        if self._grayscale:
            self._num_channels = 1
        else:
            self._num_channels = self._video_info['num_channels']

        if self._single_precision:
            self._type = np.dtype(np.float32)
        else:
            self._type = self._video_info['data_type']

        if crop is not None:
            self._width = int(crop[2] * resize)
            self._height = int(crop[3] * resize)
        else:
            self._width = self._video_info['width'] * resize
            self._height = self._video_info['height'] * resize
            
        # compute number of frames
        self._num_buffered_frames = min(
            self._number_of_frames,
            int(self._memsize_bytes // (self._width*self._height*self._num_channels*self._type.itemsize))
        )
        
        # Turns out it's faster to store this in a list rather than
        # in a large preallocated 3D or 4D numpy array (at least 
        # when stacking in the last dimension).
        # It's also faster if I subsequently modify the image 
        # cause I don't need to copy
        self._mem_buffer = [] 

        # open video capture
        if backend is not None:
            self._capture = cv2.VideoCapture(filename, backend)
        else:
            self._capture = cv2.VideoCapture(filename)
        self.open = True

        # load frames from file into memory
        print('Buffering video...')
        for i in tqdm(range(self._num_buffered_frames)): 
            rval, frame = self.read()
            if not rval:
                raise RuntimeError('Unable to buffer all frames')
            
            if self._grayscale:
                frame = im2gray(frame)

            if self._single_precision:
                frame = im2single(frame)

            self._mem_buffer.append(frame)

        # close video capture
        self._capture.release()
        self.open = True

    def close(self):
        self.open = False
    
    def is_open(self) -> bool:
        return self.open
    
    def reset_reader(self) -> None:
        self._current_frame = 0

    def next_frame(self) -> Tuple[bool, Optional[NDArray]]:
        if self._current_frame < self._num_buffered_frames:
            ret, frame = (True, self._mem_buffer[self._current_frame])
            self._current_frame += 1
        else:
            ret, frame = (False, None)
        return (ret, frame)

    def read(self) -> Tuple[bool,NDArray]:
        """
        TODO
        """
        rval, frame = self._capture.read()
        if rval:
            if self._crop is not None:
                frame = frame[
                    self._crop[1]:self._crop[1]+self._crop[3],
                    self._crop[0]:self._crop[0]+self._crop[2]
                ]
            if (self._resize is not None) and (self._resize != 1):
                frame = cv2.resize(
                    frame,
                    None,
                    None,
                    self._resize,
                    self._resize,
                    cv2.INTER_NEAREST
                )
        return (rval, frame)
    
    def get_fps(self) -> float:
        return self._fps
    
    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height
    
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_type(self) -> np.dtype:
        return self._type
    
    def get_current_frame_index(self) -> int:
        return self._current_frame

    def get_filename(self) -> str:
        return self._filename
    
    def get_number_of_frame(self) -> int:
        return self._num_buffered_frames
    