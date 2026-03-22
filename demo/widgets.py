"""Reusable Qt widgets for the Meeting Assistant UI."""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QTextBrowser, QSizePolicy, QWidget,
    QComboBox, QPushButton, QLineEdit, QFileDialog,
    QRadioButton, QButtonGroup,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont


class AudioInputWidget(QFrame):
    """Card widget for choosing audio input source (microphone or file)."""

    # Emitted when user picks a file via the browse dialog
    file_selected = Signal(str)
    # Emitted when the input mode changes: "microphone", "system" or "file"
    mode_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("audioInputFrame")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        header.setSpacing(8)
        icon_label = QLabel("🎧")
        icon_label.setStyleSheet("font-size: 16px;")
        title_label = QLabel("Audio Input")
        title_label.setObjectName("panelTitle")
        header.addWidget(icon_label)
        header.addWidget(title_label)
        header.addStretch()
        layout.addLayout(header)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #2a2d3a; max-height: 1px; border: none;")
        layout.addWidget(sep)

        # ── Mode selection (radio buttons) ──
        mode_row = QHBoxLayout()
        mode_row.setSpacing(16)

        self._btn_group = QButtonGroup(self)

        self.radio_mic = QRadioButton("🎤  Microphone")
        self.radio_mic.setObjectName("radioOption")
        self.radio_mic.setChecked(True)

        self.radio_system = QRadioButton("🔊  System Audio")
        self.radio_system.setObjectName("radioOption")

        self.radio_file = QRadioButton("📁  Audio File")
        self.radio_file.setObjectName("radioOption")

        self._btn_group.addButton(self.radio_mic, 0)
        self._btn_group.addButton(self.radio_system, 1)
        self._btn_group.addButton(self.radio_file, 2)

        mode_row.addWidget(self.radio_mic)
        mode_row.addWidget(self.radio_system)
        mode_row.addWidget(self.radio_file)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ── Microphone device selector ──
        self.mic_row = QHBoxLayout()
        self.mic_row.setSpacing(8)
        mic_label = QLabel("Device:")
        mic_label.setObjectName("inputLabel")
        self.mic_combo = QComboBox()
        self.mic_combo.setObjectName("audioCombo")
        self.mic_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_mic_devices()
        self.mic_row.addWidget(mic_label)
        self.mic_row.addWidget(self.mic_combo)

        self.mic_container = QWidget()
        self.mic_container.setLayout(self.mic_row)
        layout.addWidget(self.mic_container)

        # ── System audio (loopback) device selector ──
        self.sys_row = QHBoxLayout()
        self.sys_row.setSpacing(8)
        sys_label = QLabel("Output device:")
        sys_label.setObjectName("inputLabel")
        sys_label.setMinimumWidth(90)
        self.sys_combo = QComboBox()
        self.sys_combo.setObjectName("audioCombo")
        self.sys_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_loopback_devices()
        self.sys_row.addWidget(sys_label)
        self.sys_row.addWidget(self.sys_combo)

        self.sys_container = QWidget()
        self.sys_container.setLayout(self.sys_row)
        self.sys_container.setVisible(False)
        layout.addWidget(self.sys_container)

        # ── File chooser row ──
        self.file_row = QHBoxLayout()
        self.file_row.setSpacing(8)
        file_label = QLabel("File:")
        file_label.setObjectName("inputLabel")
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setObjectName("filePathEdit")
        self.file_path_edit.setPlaceholderText("Select an audio file (.wav, .mp3, .flac)...")
        self.file_path_edit.setReadOnly(True)
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.setObjectName("btnBrowse")

        self.file_row.addWidget(file_label)
        self.file_row.addWidget(self.file_path_edit, stretch=1)
        self.file_row.addWidget(self.btn_browse)

        self.file_container = QWidget()
        self.file_container.setLayout(self.file_row)
        self.file_container.setVisible(False)
        layout.addWidget(self.file_container)

        # ── Internal connections ──
        self._btn_group.idToggled.connect(self._on_mode_toggled)
        self.btn_browse.clicked.connect(self._on_browse)

    # ── Public API ───────────────────────────────────────────────────────

    @property
    def mode(self) -> str:
        """Return 'microphone', 'system' or 'file'."""
        if self.radio_mic.isChecked():
            return "microphone"
        if self.radio_system.isChecked():
            return "system"
        return "file"

    @property
    def selected_file(self) -> str:
        return self.file_path_edit.text()

    @property
    def selected_device_index(self) -> int | None:
        """Return the sounddevice device index for mic mode, or None for default."""
        return self.mic_combo.currentData()

    @property
    def selected_loopback_device_index(self) -> int | None:
        """Return the sounddevice device index for system/loopback mode."""
        return self.sys_combo.currentData()

    # ── Private ──────────────────────────────────────────────────────────

    def _populate_mic_devices(self):
        self.mic_combo.addItem("Default microphone", None)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    name = dev["name"]
                    self.mic_combo.addItem(f"{name}  (#{i})", i)
        except Exception:
            pass

    def _populate_loopback_devices(self):
        """Populate combo with output/loopback devices (WASAPI on Windows)."""
        self.sys_combo.addItem("Default output (loopback)", None)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev["max_output_channels"] > 0:
                    name = dev["name"]
                    hostapi = sd.query_hostapis(dev["hostapi"])["name"]
                    self.sys_combo.addItem(f"{name} [{hostapi}]  (#{i})", i)
        except Exception:
            pass

    def _on_mode_toggled(self, _id: int, checked: bool):
        if not checked:
            return
        current = self.mode
        self.mic_container.setVisible(current == "microphone")
        self.sys_container.setVisible(current == "system")
        self.file_container.setVisible(current == "file")
        self.mode_changed.emit(current)

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        )
        if path:
            self.file_path_edit.setText(path)
            self.file_selected.emit(path)


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
