# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GooseReplay.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1600, 910)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 1600, 900))
        self.label.setLineWidth(0)
        self.label.setText("")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(440, 800, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(590, 800, 75, 23))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(890, 800, 75, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(1030, 800, 75, 23))
        self.pushButton_4.setObjectName("pushButton_4")
        self.index_represent = QtWidgets.QLabel(self.centralwidget)
        self.index_represent.setGeometry(QtCore.QRect(680, 830, 191, 21))
        self.index_represent.setStyleSheet("background-color:white")
        self.index_represent.setAlignment(QtCore.Qt.AlignCenter)
        self.index_represent.setObjectName("index_represent")
        self.latest = QtWidgets.QPushButton(self.centralwidget)
        self.latest.setGeometry(QtCore.QRect(1170, 800, 75, 23))
        self.latest.setObjectName("latest")
        self.oldest = QtWidgets.QPushButton(self.centralwidget)
        self.oldest.setGeometry(QtCore.QRect(300, 800, 75, 23))
        self.oldest.setObjectName("oldest")
        self.clear = QtWidgets.QPushButton(self.centralwidget)
        self.clear.setGeometry(QtCore.QRect(740, 800, 75, 23))
        self.clear.setObjectName("clear")
        self.pause = QtWidgets.QPushButton(self.centralwidget)
        self.pause.setGeometry(QtCore.QRect(900, 830, 75, 23))
        self.pause.setText("")
        self.pause.setObjectName("pause")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "后退 5s"))
        self.pushButton_2.setText(_translate("MainWindow", "后退 1s"))
        self.pushButton_3.setText(_translate("MainWindow", "前进 1s"))
        self.pushButton_4.setText(_translate("MainWindow", "前进 5s"))
        self.index_represent.setText(_translate("MainWindow", "打开GooseGooseDuck，开杀！"))
        self.latest.setText(_translate("MainWindow", "最新"))
        self.oldest.setText(_translate("MainWindow", "最旧"))
        self.clear.setText(_translate("MainWindow", "清空"))