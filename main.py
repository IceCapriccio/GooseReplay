from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import PyQt5
import sys
from PyQt5.QtGui import QIcon
from PyQt5 import QtGui
from typing import List
from GooseReplayUI import Ui_MainWindow
# pyuic5 GooseReplay.ui -o GooseReplayUI.py
# pyinstaller  -D -w main.py -p GooseReplayUI.py

image_list = []





class GooseReplay(QMainWindow, Ui_MainWindow):
    def __init__(self, background, image_list):
        # 使用父类的构造函数，即初始化列表
        super(GooseReplay, self).__init__()
        self.setupUi(self)
        self.cur = -1
        self.recording = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.grab)
        self.timer.start(1000)

        self.pushButton.clicked.connect(self.backward5)
        self.pushButton_2.clicked.connect(self.backward1)
        self.pushButton_3.clicked.connect(self.forward1)
        self.pushButton_4.clicked.connect(self.forward5)
        self.latest.clicked.connect(self.to_latest)
        self.oldest.clicked.connect(self.to_oldest)
        self.clear.clicked.connect(self.clear_images)
        self.pause.clicked.connect(self.pause_resume)
        self.pause.setText('点击暂停')

        self.image_list = image_list

    def pause_resume(self):
        if self.recording:
            self.timer.stop()
            self.recording = False
            self.pause.setText('点击开始')
        else:
            self.timer.start()
            self.recording = True
            self.pause.setText('点击暂停')

    def update_index(self):
        if len(self.image_list) == 0:
            return
        self.index_represent.setText(f'此图距当前时间第{self.cur + 1}近，共{len(self.image_list)}张')

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

    def to_latest(self):
        if len(self.image_list) == 0:
            return
        self.cur = 0
        self.show_picture()
        self.update_index()

    def to_oldest(self):
        if len(self.image_list) == 0:
            return
        self.cur = len(self.image_list) - 1
        self.show_picture()
        self.update_index()

    def forward5(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur - 5 < len(self.image_list):
            self.cur -= 5
        else:
            self.to_latest()
        self.show_picture()
        self.update_index()

    def forward1(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur - 1 < len(self.image_list):
            self.cur -= 1
        else:
            self.to_latest()
        self.show_picture()
        self.update_index()

    def backward1(self):
        if len(self.image_list) == 0:
            return
        if 0 <= self.cur + 1 < len(self.image_list):
            self.cur += 1
        else:
            self.to_oldest()
        self.show_picture()
        self.update_index()

    def backward5(self):
        if len(self.image_list) == 0:
            return
        if self.cur + 5 < len(self.image_list):
            self.cur += 5
        else:
            self.to_oldest()
        self.show_picture()
        self.update_index()

    def clear_images(self):
        self.cur = -1
        self.image_list[:] = []
        self.label.clear()
        self.index_represent.setText('将GooseGooseDuck置于顶层以录制')

    def grab(self):
        import time

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
            import win32gui
            import time
            from PIL import ImageGrab
            (x1, y1, x2, y2), handle = get_window_pos('Goose Goose Duck')
            if win32gui.GetForegroundWindow() != handle:
                return
            time.sleep(0.5)
            # 截图
            grab_image = ImageGrab.grab((x1, y1, x2, y2))
            return grab_image
        # print('grab')
        max_images = 180
        if (image := fetch_image()) is not None:
            image_list.insert(0, (time.time(), image))
            if len(image_list) > max_images:
                image_list.pop()

        # time.sleep(0.5)
        # print([t for t, image in image_list])


if __name__ == '__main__':
    # 获取系统参数列表
    app = QApplication(sys.argv)

    # 创建实体对象
    main = GooseReplay(None, image_list)
    # 显示窗体
    main.show()

    # 进入主循环，安全退出程序
    sys.exit(app.exec_())
