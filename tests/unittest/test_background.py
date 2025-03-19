import numpy as np
import pytest
from video_tools import BackgroundImage

@pytest.fixture
def background_subtractor():
    """Fixture to initialize a BackgroundImage object."""
    bg_image = np.random.randint(0, 255, (3, 3), dtype=np.uint8)
    np.save("test_bg.npy", bg_image)  # Save a test background image
    subtractor = BackgroundImage("test_bg.npy")
    subtractor.initialize()
    return subtractor


def test_subtract_background(background_subtractor):
    """Test that subtract_background produces expected output."""
    test_image = np.random.randint(0, 255, (3, 3), dtype=np.uint8)
    result = background_subtractor.subtract_background(test_image)

    assert result.shape == (3, 3)
    assert result.dtype == np.float32
    assert np.all(result >= 0)  # Ensure non-negative values after max()