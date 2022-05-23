import threading

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

    def print_and_log(self, text):
        """Utility used to both print to terminal and log text to a QTextEdit
         item in a thread-safe manner. Use only for important user info.

        Args:
            text (str): Text to be printed and logged

        """
        self.lock.acquire()
        try:
            print(text)
            # causes issue if you clik on terminal (tied to CMD QuickEdit mode)
            self.moveCursor(QTextCursor.End)
            self.insertPlainText(f"\n{text}")
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )
        finally:
            self.lock.release()
