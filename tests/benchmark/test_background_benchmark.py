import numpy as np
import pytest
from video_tools import BackgroundImage  

@pytest.fixture
def background_subtractor():
    bg_image = np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)
    np.save("test_bg.npy", bg_image)
    subtractor = BackgroundImage("test_bg.npy")
    subtractor.initialize()
    return subtractor

@pytest.fixture
def test_image():
    return np.random.randint(0, 256, (1000, 1000), dtype=np.uint8)

def test_benchmark_subtract_background(benchmark, background_subtractor, test_image):
    benchmark(background_subtractor.subtract_background, test_image)

def test_benchmark_subtract_background2(benchmark, background_subtractor, test_image):
    benchmark(background_subtractor.subtract_background2, test_image)