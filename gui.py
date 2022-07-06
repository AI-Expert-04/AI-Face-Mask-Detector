from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *

import os
import data
import main


class LoadDataWorker(QThread):
    finished = pyqtSignal()

    def run(self):
        if not os.path.exists('data/without_mask'):
            data.download_image('without_mask')
        if not os.path.exists('data/with_mask'):
            data.generate_data()
        self.finished.emit()


class LoadVideoWorker(QThread):
    finished = pyqtSignal()

    def set_video_path(self, video_path):
        self.video_path = video_path

    def run(self):
        main.video_processing(self.video_path, True)
        self.finished.emit()


class FaceMaskDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        centralWidget = QWidget()

        self.setWindowTitle('Face Mask Detector')
        self.setGeometry(0, 0, 400, 300)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        video_widget = QVideoWidget()

        self.load_data_button = QPushButton('데이터 로드')
        self.load_data_button.clicked.connect(self.load_data)

        self.load_data_worker = LoadDataWorker()
        self.load_data_worker.finished.connect(self.load_data_finished)

        self.load_video_button = QPushButton('영상 열기')
        self.load_video_button.setEnabled(False)
        self.load_video_button.clicked.connect(self.load_video)

        self.load_video_worker = LoadVideoWorker()
        self.load_video_worker.finished.connect(self.load_video_finished)

        self.play_button = QPushButton()
        self.play_button.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.set_position)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.position_slider)

        loader_layout = QHBoxLayout()
        loader_layout.addWidget(self.load_data_button)
        loader_layout.addWidget(self.load_video_button)

        layout = QVBoxLayout()
        layout.addWidget(video_widget)
        layout.addLayout(control_layout)
        layout.addLayout(loader_layout)

        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        self.media_player.setVideoOutput(video_widget)
        self.media_player.stateChanged.connect(self.media_state_changed)
        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)
        self.media_player.error.connect(self.handle_error)

        self.show()

    def load_data(self):
        self.load_data_button.setEnabled(False)
        self.load_data_button.setText('데이터 로드 중')
        self.load_data_worker.start()

    def load_data_finished(self):
        self.load_data_button.setText('데이터 로드 완료')
        self.load_video_button.setEnabled(True)

    def load_video(self):
        video_path, _ = QFileDialog.getOpenFileName(self, '', ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi *.wmv)")
        if video_path != '':
            self.load_video_worker.set_video_path(video_path)
        self.load_video_button.setEnabled(False)
        self.load_video_button.setText('영상 처리 중')
        self.load_video_worker.start()

    def load_video_finished(self):
        self.load_video_button.setText('영상 처리 완료')
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(QFileInfo('outputs/output.wmv').absoluteFilePath())))
        self.play_button.setEnabled(True)

    def play_video(self):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.media_player.pause()
        else:
            self.media_player.play()

    def media_state_changed(self, state):
        if self.media_player.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def position_changed(self, position):
        self.position_slider.setValue(position)

    def duration_changed(self, duration):
        self.position_slider.setRange(0, duration)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def handle_error(self):
        print("Error: " + self.media_player.errorString())


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    player = FaceMaskDetector()
    sys.exit(app.exec_())
