# widget to specify a background subtraction method

from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QStackedWidget, QLabel, QVBoxLayout, QWidget
from .video_reader import OpenCV_VideoReader
from .background import Polarity, BackgroundSubtractor, InpaintBackground, NoBackgroundSub, BackgroundImage, StaticBackground, DynamicBackground, DynamicBackgroundMP
from qt_widgets import (
    LabeledSpinBox, LabeledComboBox, FileOpenLabeledEditButton,
    FileSaveLabeledEditButton, NDarray_to_QPixmap
)
import os
import cv2
import numpy as np
from numpy.typing import NDArray

# TODO add the possibility to supply a video reader

class BackgroundSubtractorWidget(QWidget):

    background_initialized = pyqtSignal(int,int,bytes)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.background_subtractor = None
        self.video_file = None
        self.declare_components()
        self.layout_components()
        self.update_background_subtractor()

        self.timer = QTimer()
        self.timer.timeout.connect(self.main)
        self.timer.setInterval(16)
        self.timer.start()

    def declare_components(self):

        # none
        self.parameters_none = QWidget()
        self.none_height = LabeledSpinBox()
        self.none_height.setText('Height')
        self.none_height.setRange(0,10000000)
        self.none_height.setValue(0)
        self.none_height.valueChanged.connect(self.update_background_subtractor)
        self.none_width = LabeledSpinBox()
        self.none_width.setText('Width')
        self.none_width.setRange(0,10000000)
        self.none_width.setValue(0)
        self.none_width.valueChanged.connect(self.update_background_subtractor)

        # image background
        self.parameters_image = QWidget()
        self.image_filename = FileOpenLabeledEditButton()
        self.image_filename.textChanged.connect(self.update_background_subtractor)

        # inpaint background
        self.parameters_inpaint = QWidget()
        self.inpaint_filename = FileOpenLabeledEditButton()
        self.inpaint_filename.textChanged.connect(self.update_background_subtractor)
        self.inpaint_frame_num = LabeledSpinBox()
        self.inpaint_frame_num.setText('Frame num.')
        self.inpaint_frame_num.setRange(0,10000000)
        self.inpaint_frame_num.setValue(0)
        self.inpaint_frame_num.valueChanged.connect(self.update_background_subtractor)
        self.inpaint_radius = LabeledSpinBox()
        self.inpaint_radius.setText('Radius')
        self.inpaint_radius.setRange(0,100)
        self.inpaint_radius.setValue(3)
        self.inpaint_radius.valueChanged.connect(self.update_background_subtractor)
        self.inpaint_algo = LabeledComboBox(self)
        self.inpaint_algo.setText('algorithm')
        self.inpaint_algo.addItem('navier-stokes')
        self.inpaint_algo.addItem('telea')
        self.inpaint_algo.currentIndexChanged.connect(self.update_background_subtractor)

        # static background
        self.parameters_static = QWidget()
        self.static_filename = FileOpenLabeledEditButton()
        self.static_filename.textChanged.connect(self.update_background_subtractor)
        self.static_numsamples = LabeledSpinBox()
        self.static_numsamples.setText('Number images')
        self.static_numsamples.setRange(1,10000)
        self.static_numsamples.setValue(10)
        self.static_numsamples.valueChanged.connect(self.update_background_subtractor)

        # dynamic background    
        self.parameters_dynamic = QWidget()
        self.dynamic_numsamples = LabeledSpinBox()
        self.dynamic_numsamples.setText('Number images')
        self.dynamic_numsamples.setRange(1,10000)
        self.dynamic_numsamples.setValue(10)
        self.dynamic_numsamples.valueChanged.connect(self.update_background_subtractor)
        self.dynamic_samplefreq = LabeledSpinBox()
        self.dynamic_samplefreq.setText('Frequency')
        self.dynamic_samplefreq.setRange(1,10000)
        self.dynamic_samplefreq.setValue(10)
        self.dynamic_samplefreq.valueChanged.connect(self.update_background_subtractor)
        
        # dynamic multiprocessing
        self.parameters_dynamic_mp = QWidget()
        self.dynamic_mp_numsamples = LabeledSpinBox()
        self.dynamic_mp_numsamples.setText('Number images')
        self.dynamic_mp_numsamples.setRange(1,10000)
        self.dynamic_mp_numsamples.setValue(10)
        self.dynamic_mp_numsamples.valueChanged.connect(self.update_background_subtractor)
        self.dynamic_mp_samplefreq = LabeledSpinBox()
        self.dynamic_mp_samplefreq.setText('Frequency')
        self.dynamic_mp_samplefreq.setRange(1,10000)
        self.dynamic_mp_samplefreq.setValue(10)
        self.dynamic_mp_samplefreq.valueChanged.connect(self.update_background_subtractor)
        self.dynamic_mp_width = LabeledSpinBox()
        self.dynamic_mp_width.setText('width')
        self.dynamic_mp_width.setRange(1,10000)
        self.dynamic_mp_width.setValue(10)
        self.dynamic_mp_width.valueChanged.connect(self.update_background_subtractor)
        self.dynamic_mp_height = LabeledSpinBox()
        self.dynamic_mp_height.setText('height')
        self.dynamic_mp_height.setRange(1,10000)
        self.dynamic_mp_height.setValue(10)
        self.dynamic_mp_height.valueChanged.connect(self.update_background_subtractor)
        
        # drop-down list to choose the background subtraction method
        self.bckgsub_method_combobox = LabeledComboBox(self)
        self.bckgsub_method_combobox.setText('method')
        self.bckgsub_method_combobox.addItem('none')
        self.bckgsub_method_combobox.addItem('image')
        self.bckgsub_method_combobox.addItem('inpaint')
        self.bckgsub_method_combobox.addItem('static')
        self.bckgsub_method_combobox.addItem('dynamic')
        self.bckgsub_method_combobox.addItem('dynamic mp')
        self.bckgsub_method_combobox.currentIndexChanged.connect(self.on_method_change)

        self.init_button = QPushButton('initialize', self)
        self.init_button.clicked.connect(self.initialize_background_subtractor)

        self.bckgsub_polarity_combobox = LabeledComboBox(self)
        self.bckgsub_polarity_combobox.setText('polarity')
        self.bckgsub_polarity_combobox.addItem('dark on bright')
        self.bckgsub_polarity_combobox.addItem('bright on dark')
        self.bckgsub_polarity_combobox.currentIndexChanged.connect(self.on_polarity_change)

        self.bckgsub_parameter_stack = QStackedWidget(self)
        self.bckgsub_parameter_stack.addWidget(self.parameters_none)
        self.bckgsub_parameter_stack.addWidget(self.parameters_image)
        self.bckgsub_parameter_stack.addWidget(self.parameters_inpaint)
        self.bckgsub_parameter_stack.addWidget(self.parameters_static)
        self.bckgsub_parameter_stack.addWidget(self.parameters_dynamic)
        self.bckgsub_parameter_stack.addWidget(self.parameters_dynamic_mp)

        self.save_filename = FileSaveLabeledEditButton()
        self.save_filename.setText('Save background image:')
        self.save_filename.textChanged.connect(self.save_background_image)

        self.zoom = LabeledSpinBox(self)
        self.zoom.setText('zoom (%)')
        self.zoom.setRange(25,500)
        self.zoom.setValue(50)
        self.zoom.setSingleStep(25)

        self.background_image = QLabel(self)

    def layout_components(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.bckgsub_method_combobox)
        main_layout.addWidget(self.bckgsub_polarity_combobox)
        main_layout.addWidget(self.bckgsub_parameter_stack)
        main_layout.addWidget(self.init_button)
        main_layout.addWidget(self.zoom)
        main_layout.addWidget(self.background_image)
        main_layout.addWidget(self.save_filename)

        none_layout = QVBoxLayout(self.parameters_none)
        none_layout.addWidget(self.none_height)
        none_layout.addWidget(self.none_width)
        none_layout.addStretch()

        image_layout = QVBoxLayout(self.parameters_image)
        image_layout.addWidget(self.image_filename)
        image_layout.addStretch()

        inpaint_layout = QVBoxLayout(self.parameters_inpaint)
        inpaint_layout.addWidget(self.inpaint_filename)
        inpaint_layout.addWidget(self.inpaint_frame_num)
        inpaint_layout.addWidget(self.inpaint_radius)
        inpaint_layout.addWidget(self.inpaint_algo)
        inpaint_layout.addStretch()

        static_layout = QVBoxLayout(self.parameters_static)
        static_layout.addWidget(self.static_filename)
        static_layout.addWidget(self.static_numsamples)
        static_layout.addStretch()

        dynamic_layout = QVBoxLayout(self.parameters_dynamic)
        dynamic_layout.addWidget(self.dynamic_numsamples)
        dynamic_layout.addWidget(self.dynamic_samplefreq)
        dynamic_layout.addStretch()

        dynamic_mp_layout = QVBoxLayout(self.parameters_dynamic_mp)
        dynamic_mp_layout.addWidget(self.dynamic_mp_numsamples)
        dynamic_mp_layout.addWidget(self.dynamic_mp_samplefreq)
        dynamic_mp_layout.addWidget(self.dynamic_mp_height)
        dynamic_mp_layout.addWidget(self.dynamic_mp_width)

    def on_method_change(self, index):
        self.bckgsub_parameter_stack.setCurrentIndex(index)
        self.update_background_subtractor()

    def on_polarity_change(self, index):
        if self.background_subtractor is not None:
            if index ==0:
                self.background_subtractor.set_polarity(Polarity.DARK_ON_BRIGHT)
            else:
                self.background_subtractor.set_polarity(Polarity.BRIGHT_ON_DARK)

    def set_video_file(self, filename: str) -> None:
        self.video_file = filename
        self.static_filename.setEditField(filename)
        self.inpaint_filename.setEditField(filename)

    def update_background_subtractor(self):
        method = self.bckgsub_method_combobox.currentIndex()

        if self.bckgsub_polarity_combobox.currentIndex() == 0:
            polarity = Polarity.DARK_ON_BRIGHT
        else:
            polarity = Polarity.BRIGHT_ON_DARK
        
        if method == 0:
            self.background_subtractor = NoBackgroundSub(
                height = self.none_height.value(),
                width = self.none_width.value(),
                polarity = polarity
            )

        if method == 1:
            filepath = self.image_filename.text()
            if os.path.exists(filepath):
                self.background_subtractor = BackgroundImage(
                    image_file_name = filepath,
                    polarity = polarity
                )

        if method == 2:
            filepath = self.inpaint_filename.text()
            if self.inpaint_algo.currentIndex() == 0:
                algo = cv2.INPAINT_NS
            else:
                algo = cv2.INPAINT_TELEA

            if os.path.exists(filepath):
                video_reader = OpenCV_VideoReader()
                video_reader.open_file(filepath)

                self.background_subtractor = InpaintBackground(
                    video_reader = video_reader,
                    frame_num = self.inpaint_frame_num.value(),
                    inpaint_radius = self.inpaint_radius.value(),
                    algo = algo,
                    polarity = polarity
                )

        if method == 3:
            filepath = self.static_filename.text()
            if os.path.exists(filepath):
                video_reader = OpenCV_VideoReader()
                video_reader.open_file(filepath)
                self.background_subtractor = StaticBackground(
                    video_reader = video_reader,
                    num_sample_frames = self.static_numsamples.value(),
                    polarity = polarity
                )
            
        if method == 4:
            self.background_subtractor = DynamicBackground(
                num_sample_frames = self.dynamic_numsamples.value(),
                sample_every_n_frames = self.dynamic_samplefreq.value(),
                polarity = polarity
            )

        if method == 5:
            self.background_subtractor = DynamicBackgroundMP(
                num_images = self.dynamic_mp_numsamples.value(),
                every_n_image = self.dynamic_mp_samplefreq.value(),
                width = self.dynamic_mp_width.value(),
                height = self.dynamic_mp_height.value(),
                polarity = polarity
            )

    def save_background_image(self, filename):
        if self.background_subtractor is not None:
            bckg_image = self.background_subtractor.get_background_image()
            if bckg_image is not None:
                bckg_image = (255*bckg_image).astype(np.uint8)
                cv2.imwrite(filename, bckg_image)

    def initialize_background_subtractor(self):
        # TODO launch this in a separate thread/process otherwise the GUI becomes 
        # unresponsive
        self.background_subtractor.initialize()
        img = self.background_subtractor.get_background_image()
        self.background_initialized.emit(img.shape[0], img.shape[1], img.tobytes())

    def get_background_subtractor(self):
        return self.background_subtractor
    
    def main(self):
        # this is relying on the fact that background is copied and updated outside, that doesn't seem to 
        # to work for the multiprocessed dynamic background though
        if self.background_subtractor.is_initialized():
            image = self.background_subtractor.get_background_image() 
            if image is not None:
                image = (255*image).astype(np.uint8)
                scale = self.zoom.value()/100
                image = cv2.resize(image,None,None,scale,scale,cv2.INTER_NEAREST)
                self.background_image.setPixmap(NDarray_to_QPixmap(image))

