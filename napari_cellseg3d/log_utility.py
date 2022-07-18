import threading
import warnings

from qtpy import QtCore
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QTextEdit


class Log(QTextEdit):
    """Class to implement a log for important user info. Should be thread-safe."""

    def __init__(self, parent):
        """Creates a log with a lock for multithreading

        Args:
            parent (QWidget): parent widget to add Log instance to.
        """
        super().__init__(parent)

        # from qtpy.QtCore import QMetaType
        # parent.qRegisterMetaType<QTextCursor>("QTextCursor")

        self.lock = threading.Lock()

    # def receive_log(self, text):
    #     self.print_and_log(text)
    def write(self, message):
        self.lock.acquire()
        try:
            if not hasattr(self, "flag"):
                self.flag = False
            message = message.replace("\r", "").rstrip()
            if message:
                method = "replace_last_line" if self.flag else "append"
                QtCore.QMetaObject.invokeMethod(
                    self,
                    method,
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(str, message),
                )
                self.flag = True
            else:
                self.flag = False

        finally:
            self.lock.release()

    @QtCore.Slot(str)
    def replace_last_line(self, text):
        self.lock.acquire()
        try:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.insertBlock()
            self.setTextCursor(cursor)
            self.insertPlainText(text)
        finally:
            self.lock.release()

    def print_and_log(self, text, printing=True):
        """Utility used to both print to terminal and log text to a QTextEdit
         item in a thread-safe manner. Use only for important user info.

        Args:
            text (str): Text to be printed and logged
            printing (bool): Whether to print the message as well or not using print(). Defaults to True.

        """
        self.lock.acquire()
        try:
            if printing:
                print(text)
            # causes issue if you clik on terminal (tied to CMD QuickEdit mode on Windows)
            self.moveCursor(QTextCursor.End)
            self.insertPlainText(f"\n{text}")
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )
        finally:
            self.lock.release()

    def warn(self, warning):
        self.lock.acquire()
        try:
            warnings.warn(warning)
        finally:
            self.lock.release()
