"""Reusable Qt widgets for the Meeting Assistant UI."""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QTextBrowser, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class PanelWidget(QFrame):
    """A styled card-like container for transcript / summary panels."""

    def __init__(self, title: str, icon: str, is_summary: bool = False):
        super().__init__()
        self.setObjectName("summaryPanelFrame" if is_summary else "panelFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Panel header row
        header = QHBoxLayout()
        header.setSpacing(8)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 16px;")

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch()
        header.addWidget(self.status_label)

        layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2a2d3a; max-height: 1px; border: none;")
        layout.addWidget(sep)

        # Content area
        if is_summary:
            self.content = QTextBrowser()
            self.content.setObjectName("summaryArea")
            self.content.setOpenExternalLinks(False)
        else:
            self.content = QTextEdit()
            self.content.setObjectName("transcriptArea")
            self.content.setReadOnly(True)

        self.content.setFont(QFont("Segoe UI", 10))
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.content)

    def set_status(self, text: str, active: bool = False):
        self.status_label.setText(text)
        self.status_label.setObjectName("statusLabelActive" if active else "statusLabel")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
