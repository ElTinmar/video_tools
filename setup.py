from distutils.core import setup

setup(
    name='video_tools',
    python_requires='>=3.7',
    author='Martin Privat',
    version='0.6.6',
    packages=['video_tools','video_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='simple video reader, writer, and processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "opencv-python-headless",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@v0.9.3",
        "tqdm"
    ]
)
