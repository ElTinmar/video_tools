from distutils.core import setup

setup(
    name='video_tools',
    author='Martin Privat',
    version='0.1.4',
    packages=['video_tools','video_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='simple video reader, writer, and processing functions',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "scipy",
        "PyQt5",
        "opencv-python",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@main",
        "tqdm"
    ]
)