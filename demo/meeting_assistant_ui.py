"""
AI Meeting Assistant — Entry point.
Run this file to launch the application.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from demo.styles import STYLESHEET
from demo.main_window import MeetingAssistantWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLESHEET)

    window = MeetingAssistantWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
