"""Main application window — layout, timers, and control logic."""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSplitter, QTextEdit,
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QColor, QTextCursor, QTextCharFormat

from .widgets import PanelWidget
from .markdown_utils import markdown_to_html
from .mock_data import MOCK_TRANSCRIPT_LINES, MOCK_SUMMARIES


class MeetingAssistantWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Meeting Assistant")
        self.resize(1200, 720)
        self.setMinimumSize(800, 500)

        # ── State ──
        self._transcript_text: str = ""
        self._summary_text: str = ""
        self._transcript_idx: int = 0
        self._summary_idx: int = 0
        self._running: bool = False
        self._user_scrolled_up: bool = False

        self._build_ui()
        self._setup_timers()
        self._connect_signals()

    # ── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        root.addLayout(self._build_header())
        root.addWidget(self._build_panels(), stretch=1)
        root.addLayout(self._build_controls())

    def _build_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.setSpacing(12)

        dot = QLabel("●")
        dot.setStyleSheet("color: #7dd3fc; font-size: 24px;")

        title_col = QVBoxLayout()
        title_col.setSpacing(1)

        title = QLabel("AI Meeting Assistant")
        title.setObjectName("headerTitle")

        subtitle = QLabel("Real-time transcription & intelligent summarization")
        subtitle.setObjectName("headerSubtitle")

        title_col.addWidget(title)
        title_col.addWidget(subtitle)

        header.addWidget(dot)
        header.addLayout(title_col)
        header.addStretch()

        return header

    def _build_panels(self) -> QSplitter:
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        self.transcript_panel = PanelWidget("Transcript", "🎤", is_summary=False)
        self.summary_panel = PanelWidget("Summary", "🧠", is_summary=True)

        splitter.addWidget(self.transcript_panel)
        splitter.addWidget(self.summary_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        return splitter

    def _build_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("btnStart")

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)

        self.btn_clear = QPushButton("✕  Clear")
        self.btn_clear.setObjectName("btnClear")

        bar.addStretch()
        bar.addWidget(self.btn_start)
        bar.addWidget(self.btn_stop)
        bar.addWidget(self.btn_clear)
        bar.addStretch()

        return bar

    # ── Timers ───────────────────────────────────────────────────────────

    def _setup_timers(self):
        self._transcript_timer = QTimer(self)
        self._transcript_timer.setInterval(500)
        self._transcript_timer.timeout.connect(self._on_transcript_tick)

        self._summary_timer = QTimer(self)
        self._summary_timer.setInterval(2500)
        self._summary_timer.timeout.connect(self._on_summary_tick)

        self._highlight_timer = QTimer(self)
        self._highlight_timer.setInterval(600)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self._clear_highlight)

    # ── Signals / Slots ──────────────────────────────────────────────────

    def _connect_signals(self):
        self.btn_start.clicked.connect(self._start)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_clear.clicked.connect(self._clear)

        vbar = self.transcript_panel.content.verticalScrollBar()
        vbar.valueChanged.connect(self._on_transcript_scroll)

    # ── Control Handlers ─────────────────────────────────────────────────

    @Slot()
    def _start(self):
        if self._running:
            return
        self._running = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.transcript_panel.set_status("● Listening...", active=True)
        self.summary_panel.set_status("● Summarizing...", active=True)

        self._transcript_timer.start()
        self._summary_timer.start()

    @Slot()
    def _stop(self):
        if not self._running:
            return
        self._running = False
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self._transcript_timer.stop()
        self._summary_timer.stop()

        self.transcript_panel.set_status("Paused", active=False)
        self.summary_panel.set_status("Paused", active=False)

    @Slot()
    def _clear(self):
        self._stop()
        self._transcript_text = ""
        self._summary_text = ""
        self._transcript_idx = 0
        self._summary_idx = 0
        self.transcript_panel.content.clear()
        self.summary_panel.content.setHtml("")
        self.transcript_panel.set_status("", active=False)
        self.summary_panel.set_status("", active=False)

    # ── Mock Data Tick ───────────────────────────────────────────────────

    @Slot()
    def _on_transcript_tick(self):
        if self._transcript_idx >= len(MOCK_TRANSCRIPT_LINES):
            self._transcript_timer.stop()
            self.transcript_panel.set_status("● Completed", active=False)
            return

        line = MOCK_TRANSCRIPT_LINES[self._transcript_idx]
        self._transcript_idx += 1
        self._transcript_text += line + "\n"

        self._append_transcript_line(line)

    @Slot()
    def _on_summary_tick(self):
        if self._summary_idx >= len(MOCK_SUMMARIES):
            self._summary_timer.stop()
            self.summary_panel.set_status("● Completed", active=False)
            return

        md = MOCK_SUMMARIES[self._summary_idx]
        self._summary_idx += 1
        self._summary_text = md

        html = markdown_to_html(md)
        self.summary_panel.content.setHtml(
            f'<div style="font-family: Segoe UI, sans-serif;">{html}</div>'
        )

    # ── Transcript helpers ───────────────────────────────────────────────

    def _append_transcript_line(self, line: str):
        editor: QTextEdit = self.transcript_panel.content
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#7dd3fc"))
        cursor.insertText(line + "\n", fmt)

        if not self._user_scrolled_up:
            editor.setTextCursor(cursor)
            editor.ensureCursorVisible()

        self._highlight_timer.start()

    def _clear_highlight(self):
        editor: QTextEdit = self.transcript_panel.content
        cursor = editor.textCursor()
        cursor.select(QTextCursor.Document)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor("#c9d1d9"))
        cursor.mergeCharFormat(fmt)

    @Slot(int)
    def _on_transcript_scroll(self, value: int):
        vbar = self.transcript_panel.content.verticalScrollBar()
        self._user_scrolled_up = value < vbar.maximum()
