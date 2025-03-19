from video_tools import (
    InMemory_OpenCV_VideoReader, Polarity,
    NoBackgroundSub, BackgroundImage, InpaintBackground, 
    StaticBackground, DynamicBackground, DynamicBackgroundMP
)
from image_tools import im2single, im2single_GPU, im2gray
from tqdm import tqdm
import cProfile
import pstats

DATA = [
    ('../toy_data/multi_freelyswimming_1800x1800px.avi', '../toy_data/multi_freelyswimming_1800x1800px.png', Polarity.BRIGHT_ON_DARK, 40),
    ('../toy_data/single_freelyswimming_504x500px.avi', '../toy_data/single_freelyswimming_504x500px.png', Polarity.DARK_ON_BRIGHT, 40),
    ('../toy_data/single_headembedded_544x380px_noparam.avi', '../toy_data/single_headembedded_544x380px_noparam.png', Polarity.DARK_ON_BRIGHT, 100),
    ('../toy_data/single_headembedded_544x380px_param.avi', '../toy_data/single_headembedded_544x380px_param.png', Polarity.DARK_ON_BRIGHT, 100)
]

INPUT_VIDEO, BACKGROUND_IMAGE, POLARITY, PIX_PER_MM = DATA[0]

# video reader
video_reader = InMemory_OpenCV_VideoReader()
video_reader.open_file(
    filename=INPUT_VIDEO,
    safe=False,
    memsize_bytes=4e9,
    single_precision=False,
    grayscale=False
)
num_frames = video_reader.get_number_of_frame()

# background subtraction
background_sub = BackgroundImage(
    polarity = POLARITY,
    image_file_name = BACKGROUND_IMAGE,
    use_gpu=False
)
background_sub.initialize()

with cProfile.Profile() as pr:

    print('Background subtraction ...')
    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # background sub
        background_sub.subtract_background(frame)

video_reader.close()
ps = pstats.Stats(pr)
ps.dump_stats('raw_speed.prof')