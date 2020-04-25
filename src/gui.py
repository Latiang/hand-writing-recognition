import sys

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QHBoxLayout, QPushButton
from PyQt5 import QtCore

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('QHBoxLayout')
        layout = QHBoxLayout()
        layout.addWidget(QPushButton('Left'))
        layout.addWidget(QPushButton('Center'))
        layout.addWidget(QPushButton('Right'))
        test_button = QPushButton('Test')
        test_button.clicked.connect(lambda x: self.test("wooo"))
        layout.addWidget(test_button)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        print(event.key())
        event.accept()

    def test(self, message = "Test"):
        print("Test test: ", message)

class MainApplication(QApplication):
    def __init__(self, dataset):
        super().__init__([])

        window = MainWindow()

        window.show()
        sys.exit(self.exec_())