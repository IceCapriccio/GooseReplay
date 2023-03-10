import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import PyQt5
import sys
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
from typing import List
from GooseReplayUI import Ui_MainWindow
import win32gui
import time
from PIL import ImageGrab
from auto_stop import GooseStopper
from config import *
# pyuic5 GooseReplay.ui -o GooseReplayUI.py
# pyinstaller -D -w main.py -p GooseReplayUI.py


class GooseReplay(QMainWindow, Ui_MainWindow):
    def __init__(self):
        # 使用父类的构造函数，即初始化列表
        super(GooseReplay, self).__init__()
        self.setupUi(self)
        self.cur = -1
        self.recording = True
        self.neutron = torch.load('goose_stopper_1.0.th')

        self.eye = QTimer(self)
        self.eye.timeout.connect(self.grab_frame)
        self.eye.start(1000)

        self.brain = QTimer(self)
        self.brain.timeout.connect(self.recognize)
        self.brain.start(5000)

        self.pushButton.clicked.connect(self.backward5)
        self.pushButton_2.clicked.connect(self.backward1)
        self.pushButton_3.clicked.connect(self.forward1)
        self.pushButton_4.clicked.connect(self.forward5)
        self.latest.clicked.connect(self.tolatest)
        self.oldest.clicked.connect(self.tooldest)
        self.clear.clicked.connect(self.clear_images)

        self.pause.clicked.connect(self.switch_recording)
        self.pause.setText('点击暂停')

        self.image_list = []

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        if config['zz_remind']:
            message = QMessageBox()
            message.setText('记得关闭ZZ加速器！')
            message.setWindowTitle('小提示')
            ret_val = message.exec_()

    def recognize(self):
        if len(self.image_list) == 0:
            return
        image = self.image_list[-1][1]  # get latest image
        data = np.asarray(image).transpose(2, 0, 1)[np.newaxis, ...]
        data = torch.tensor(data, dtype=torch.float)
        result = self.neutron(data).item()
        if result > 0.5 and not self.recording:  # in gaming
            self.switch_recording()
            print('brain said:"start record"')
        elif result < 0.5 and self.recording:
            self.switch_recording()
            print('brain said:"stop record"')

    def switch_recording(self):
        if self.recording:
            self.eye.stop()
            self.recording = False
            self.pause.setText('点击开始')
        else:
            self.eye.start()
            self.recording = True
            self.pause.setText('点击暂停')

    def update_index(self):
        if len(self.image_list) == 0:
            return
        self.index_represent.setText(f'此图距当前时间第{len(self.image_list) - self.cur}近，共{len(self.image_list)}张')

    def show_picture(self):
        if len(self.image_list) == 0:
            return
        im = self.image_list[self.cur][1]
        # https://stackoverflow.com/questions/34697559/pil-image-to-qpixmap-conversion-issue
        im = im.convert("RGB")
        data = im.tobytes("raw", "RGB")
        qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qim)
        self.label.setPixmap(pix)
        self.label.show()

    def tooldest(self):
        if len(self.image_list) == 0:
            return
        self.cur = 0
        self.show_picture()
        self.update_index()

    def tolatest(self):
        if len(self.image_list) == 0:
            return
        self.cur = len(self.image_list) - 1
        self.show_picture()
        self.update_index()

    def backward5(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur - 5 < len(self.image_list):
            self.cur -= 5
        else:
            self.tooldest()
        self.show_picture()
        self.update_index()

    def backward1(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur - 1 < len(self.image_list):
            self.cur -= 1
        else:
            self.tooldest()
        self.show_picture()
        self.update_index()

    def forward1(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur + 1 < len(self.image_list):
            self.cur += 1
        else:
            self.tolatest()
        self.show_picture()
        self.update_index()

    def forward5(self):
        if len(self.image_list) == 0:
            return
        if self.cur + 5 < len(self.image_list):
            self.cur += 5
        else:
            self.tolatest()
        self.show_picture()
        self.update_index()

    def clear_images(self):
        self.cur = -1
        self.image_list.clear()
        self.label.clear()
        self.index_represent.setText('将GooseGooseDuck置于顶层以录制')

    def grab_frame(self):
        def get_window_pos(name):
            import win32gui
            name = name
            handle = win32gui.FindWindow(0, name)
            # 获取窗口句柄
            if handle == 0:
                return (0, 0, 0, 0), handle
            else:
                # 返回坐标值和handle
                return win32gui.GetWindowRect(handle), handle

        def fetch_image():
            (x1, y1, x2, y2), handle = get_window_pos('Goose Goose Duck')
            if win32gui.GetForegroundWindow() != handle:
                return
            time.sleep(0.5)
            # 截图
            grab_image = ImageGrab.grab((x1, y1, x2, y2))
            return grab_image

        max_images = 180
        if (image := fetch_image()) is not None:
            if image.size == (1616, 936):
                image.save(f'frames/{time.strftime("%Y-%m-%d-%H-%M-%S.png", time.localtime())}', format='png')
            self.image_list.append((time.time(), image))
            if len(self.image_list) > max_images:
                self.image_list.pop(0)
                if self.cur > 0:
                    self.cur -= 1


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = GooseReplay()
    main.show()
    sys.exit(app.exec_())
