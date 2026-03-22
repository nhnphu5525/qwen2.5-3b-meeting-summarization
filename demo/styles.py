"""Global QSS stylesheet for the dark theme."""

STYLESHEET = """
QMainWindow {
    background-color: #0f1117;
}
QWidget#centralWidget {
    background-color: #0f1117;
}

/* ── Header ──────────────────────────────────── */
QLabel#headerTitle {
    color: #f0f0f0;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.5px;
    padding: 0px;
}
QLabel#headerSubtitle {
    color: #6b7280;
    font-size: 12px;
    padding: 0px;
}

/* ── Panels ──────────────────────────────────── */
QFrame#panelFrame {
    background-color: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
}
QFrame#summaryPanelFrame {
    background-color: #1e2230;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
}

/* ── Panel headers ───────────────────────────── */
QLabel#panelTitle {
    color: #e5e7eb;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
QLabel#statusLabel {
    color: #6b7280;
    font-size: 11px;
    font-weight: 500;
}
QLabel#statusLabelActive {
    color: #34d399;
    font-size: 11px;
    font-weight: 600;
}

/* ── Text areas ──────────────────────────────── */
QTextEdit#transcriptArea {
    background-color: transparent;
    color: #c9d1d9;
    border: none;
    font-size: 13.5px;
    line-height: 1.7;
    padding: 12px;
    selection-background-color: #264f78;
}
QTextBrowser#summaryArea {
    background-color: transparent;
    color: #d4d4d4;
    border: none;
    font-size: 13.5px;
    padding: 14px;
    selection-background-color: #264f78;
}

/* ── Buttons ─────────────────────────────────── */
QPushButton#btnStart {
    background-color: #22c55e;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    font-size: 13px;
    font-weight: 600;
    min-width: 90px;
}
QPushButton#btnStart:hover {
    background-color: #16a34a;
}
QPushButton#btnStart:pressed {
    background-color: #15803d;
}
QPushButton#btnStop {
    background-color: #ef4444;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 8px 24px;
    font-size: 13px;
    font-weight: 600;
    min-width: 90px;
}
QPushButton#btnStop:hover {
    background-color: #dc2626;
}
QPushButton#btnStop:pressed {
    background-color: #b91c1c;
}
QPushButton#btnClear {
    background-color: #374151;
    color: #d1d5db;
    border: 1px solid #4b5563;
    border-radius: 8px;
    padding: 8px 24px;
    font-size: 13px;
    font-weight: 600;
    min-width: 90px;
}
QPushButton#btnClear:hover {
    background-color: #4b5563;
}
QPushButton#btnClear:pressed {
    background-color: #6b7280;
}

/* ── Splitter ────────────────────────────────── */
QSplitter::handle {
    background-color: #2a2d3a;
    width: 2px;
}
QSplitter::handle:hover {
    background-color: #7dd3fc;
}

/* ── Scrollbar ───────────────────────────────── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #374151;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #4b5563;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}
"""
