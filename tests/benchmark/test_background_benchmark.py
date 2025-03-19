import numpy as np
import pytest
import os
from video_tools import BackgroundImage  

SHP = (2048, 2048)

@pytest.fixture
def background_subtractor():
    bg_image = np.random.randint(0, 255, SHP, dtype=np.uint8)
    filename = "test_bg.npy"
    np.save(filename, bg_image)
    subtractor = BackgroundImage("test_bg.npy")
    subtractor.initialize()
    yield subtractor  
    os.remove(filename)

@pytest.fixture
def test_image():
    return np.random.randint(0, 255, SHP, dtype=np.uint8)

def test_benchmark_subtract_background(benchmark, background_subtractor, test_image):
    benchmark(background_subtractor.subtract_background, test_image)

