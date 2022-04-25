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

        self.lock = threading.Lock()

    def print_and_log(self, text):
        """Utility used to both print to terminal and log text to a QTextEdit
         item in a thread-safe manner. Use only for important user info.

        Args:
            text (str): Text to be printed and logged

        """
        with self.lock:
            print(text)
            self.moveCursor(QTextCursor.End)
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum()
            )
            self.insertPlainText(f"\n{text}")


