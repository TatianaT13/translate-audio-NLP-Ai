# app.py
import os
import sys
import time
import json
import queue
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from faster_whisper import WhisperModel

from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QGroupBox,
)


# =========================
# Utils
# =========================

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def save_wav(path: str, audio: np.ndarray, sr: int):
    a = np.clip(audio, -1.0, 1.0)
    a16 = (a * 32767.0).astype(np.int16)
    wav_write(path, sr, a16)


def which_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    return exe


def ensure_ffmpeg_or_raise():
    if which_ffmpeg() is None:
        raise RuntimeError(
            "ffmpeg introuvable. Installe ffmpeg (ex: brew install ffmpeg sur macOS) "
            "pour supporter tous les formats (mp3, m4a, aac, flac, ogg, webm, etc.)."
        )


def convert_to_wav_16k_mono(src_path: str, dst_path: str) -> None:
    """
    Convertit n'importe quel format audio vers WAV 16kHz mono (PCM s16le).
    Nécessite ffmpeg.
    """
    ensure_ffmpeg_or_raise()
    cmd = [
        which_ffmpeg(),
        "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        dst_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion échouée:\n{p.stderr}")


def list_input_devices() -> List[Tuple[int, str]]:
    devices = sd.query_devices()
    res = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            res.append((i, d.get("name", "unknown")))
    return res


# =========================
# Whisper Service (shared)
# =========================

class WhisperService:
    def __init__(self):
        self._model: Optional[WhisperModel] = None
        self._model_name: Optional[str] = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"

    def load(self, model_name: str, device: str = "cpu", compute_type: Optional[str] = None):
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        if self._model is not None and self._model_name == model_name and self._device == device and self._compute_type == compute_type:
            return

        self._model_name = model_name
        self._device = device
        self._compute_type = compute_type
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_wav(self, wav_path: str, language: Optional[str], beam_size: int) -> Tuple[str, str, float]:
        assert self._model is not None, "Model not loaded"
        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=beam_size,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 350},
        )
        texts: List[str] = []
        for s in segments:
            t = (s.text or "").strip()
            if t:
                texts.append(t)
        return " ".join(texts), info.language, float(info.language_probability)


# =========================
# Workers
# =========================

class LiveWorker(QObject):
    textReady = Signal(str)
    status = Signal(str)
    error = Signal(str)
    stopped = Signal()

    def __init__(
        self,
        whisper: WhisperService,
        model_name: str,
        language: Optional[str],
        beam_size: int,
        input_device: Optional[int],
        sample_rate: int,
        chunk_seconds: float,
        min_rms: float,
        compute_type: str,
    ):
        super().__init__()
        self.whisper = whisper
        self.model_name = model_name
        self.language = language
        self.beam_size = beam_size
        self.input_device = input_device
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.min_rms = min_rms
        self.compute_type = compute_type

        self._stop = False
        self._q: "queue.Queue[np.ndarray]" = queue.Queue()

    @Slot()
    def run(self):
        try:
            self._stop = False
            sr = self.sample_rate
            chunk_samples = int(sr * self.chunk_seconds)

            self.status.emit(f"Chargement modèle {self.model_name} (cpu/{self.compute_type})")
            self.whisper.load(self.model_name, device="cpu", compute_type=self.compute_type)
            self.status.emit("Démarrage capture micro...")

            def callback(indata, frames, time_info, status):
                if self._stop:
                    return
                mono = indata[:, 0].copy()
                self._q.put(mono)

            stream_kwargs = {
                "samplerate": sr,
                "channels": 1,
                "dtype": "float32",
                "callback": callback,
            }
            if self.input_device is not None:
                stream_kwargs["device"] = self.input_device

            buffer: List[np.ndarray] = []
            buffered = 0

            with sd.InputStream(**stream_kwargs):
                while not self._stop:
                    try:
                        x = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    buffer.append(x)
                    buffered += x.shape[0]

                    if buffered < chunk_samples:
                        continue

                    allbuf = np.concatenate(buffer, axis=0)
                    chunk = allbuf[:chunk_samples]
                    rest = allbuf[chunk_samples:]
                    buffer = [rest] if rest.size else []
                    buffered = rest.shape[0] if rest.size else 0

                    level = rms(chunk)
                    if level < self.min_rms:
                        continue

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                        save_wav(tmp.name, chunk, sr)
                        t0 = time.time()
                        text, lang, prob = self.whisper.transcribe_wav(tmp.name, self.language, self.beam_size)
                        dt = time.time() - t0

                    if text:
                        stamp = time.strftime("%H:%M:%S")
                        self.textReady.emit(f"[{stamp}] ({lang},{prob:.2f}) {text}  (chunk {dt:.2f}s)")

            self.status.emit("Arrêt.")
            self.stopped.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.stopped.emit()

    def stop(self):
        self._stop = True


class FileWorker(QObject):
    textReady = Signal(str)
    status = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self,
        whisper: WhisperService,
        src_path: str,
        model_name: str,
        language: Optional[str],
        beam_size: int,
        compute_type: str,
    ):
        super().__init__()
        self.whisper = whisper
        self.src_path = src_path
        self.model_name = model_name
        self.language = language
        self.beam_size = beam_size
        self.compute_type = compute_type

    @Slot()
    def run(self):
        try:
            self.status.emit("Préparation fichier (conversion WAV 16k mono si nécessaire)...")
            self.whisper.load(self.model_name, device="cpu", compute_type=self.compute_type)

            with tempfile.TemporaryDirectory() as td:
                wav_path = os.path.join(td, "input_16k_mono.wav")
                convert_to_wav_16k_mono(self.src_path, wav_path)

                self.status.emit("Transcription en cours...")
                t0 = time.time()
                text, lang, prob = self.whisper.transcribe_wav(wav_path, self.language, self.beam_size)
                dt = time.time() - t0

            if text:
                self.textReady.emit(f"({lang},{prob:.2f}) {text}\n\nDurée transcription: {dt:.2f}s")
            else:
                self.textReady.emit("(Aucun texte détecté)")

            self.status.emit("Terminé.")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit()


# =========================
# UI
# =========================

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DEV-AI Transcriber (Live + Fichier)")
        self.resize(980, 700)

        self.whisper = WhisperService()

        self.tabs = QTabWidget()
        self.tab_live = QWidget()
        self.tab_file = QWidget()

        self.tabs.addTab(self.tab_live, "Live (micro)")
        self.tabs.addTab(self.tab_file, "Fichier")

        root = QVBoxLayout()
        root.addWidget(self.tabs)
        self.setLayout(root)

        self._build_live_tab()
        self._build_file_tab()

        self.live_thread: Optional[QThread] = None
        self.live_worker: Optional[LiveWorker] = None

        self.file_thread: Optional[QThread] = None
        self.file_worker: Optional[FileWorker] = None

        self._refresh_devices()

    # ---------- Shared controls ----------
    def _model_choices(self) -> List[str]:
        return ["small", "medium", "large-v3"]

    def _compute_choices(self) -> List[str]:
        return ["int8", "int8_float16", "float16"]

    def _language_choices(self) -> List[Tuple[str, Optional[str]]]:
        return [("Auto", None), ("Français (fr)", "fr"), ("Anglais (en)", "en")]

    def _refresh_devices(self):
        self.combo_mic.clear()
        self._mic_map = {}
        for idx, name in list_input_devices():
            label = f"{idx}: {name}"
            self.combo_mic.addItem(label)
            self._mic_map[label] = idx

    # ---------- Live tab ----------
    def _build_live_tab(self):
        layout = QVBoxLayout()
        self.tab_live.setLayout(layout)

        box = QGroupBox("Paramètres Live")
        box_l = QVBoxLayout()
        box.setLayout(box_l)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Micro:"))
        self.combo_mic = QComboBox()
        row1.addWidget(self.combo_mic, 1)
        self.btn_refresh_mic = QPushButton("Rafraîchir")
        self.btn_refresh_mic.clicked.connect(self._refresh_devices)
        row1.addWidget(self.btn_refresh_mic)
        box_l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Langue:"))
        self.combo_lang_live = QComboBox()
        for label, code in self._language_choices():
            self.combo_lang_live.addItem(label, userData=code)
        row2.addWidget(self.combo_lang_live)

        row2.addWidget(QLabel("Modèle:"))
        self.combo_model_live = QComboBox()
        for m in self._model_choices():
            self.combo_model_live.addItem(m)
        self.combo_model_live.setCurrentText("medium")
        row2.addWidget(self.combo_model_live)

        row2.addWidget(QLabel("Compute:"))
        self.combo_compute_live = QComboBox()
        for c in self._compute_choices():
            self.combo_compute_live.addItem(c)
        self.combo_compute_live.setCurrentText("int8")
        row2.addWidget(self.combo_compute_live)
        box_l.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Chunk (s):"))
        self.spin_chunk = QDoubleSpinBox()
        self.spin_chunk.setRange(1.0, 20.0)
        self.spin_chunk.setSingleStep(0.5)
        self.spin_chunk.setValue(4.0)
        row3.addWidget(self.spin_chunk)

        row3.addWidget(QLabel("Min RMS:"))
        self.spin_min_rms = QDoubleSpinBox()
        self.spin_min_rms.setDecimals(4)
        self.spin_min_rms.setRange(0.0, 0.2)
        self.spin_min_rms.setSingleStep(0.001)
        self.spin_min_rms.setValue(0.0100)
        row3.addWidget(self.spin_min_rms)

        row3.addWidget(QLabel("Beam:"))
        self.spin_beam_live = QSpinBox()
        self.spin_beam_live.setRange(1, 10)
        self.spin_beam_live.setValue(5)
        row3.addWidget(self.spin_beam_live)

        row3.addWidget(QLabel("Sample rate:"))
        self.spin_sr = QSpinBox()
        self.spin_sr.setRange(8000, 48000)
        self.spin_sr.setSingleStep(1000)
        self.spin_sr.setValue(16000)
        row3.addWidget(self.spin_sr)
        box_l.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_start_live = QPushButton("Démarrer")
        self.btn_stop_live = QPushButton("Stop")
        self.btn_stop_live.setEnabled(False)
        self.btn_clear_live = QPushButton("Effacer")
        self.btn_copy_live = QPushButton("Copier")
        self.btn_export_live = QPushButton("Exporter TXT")

        self.btn_start_live.clicked.connect(self.start_live)
        self.btn_stop_live.clicked.connect(self.stop_live)
        self.btn_clear_live.clicked.connect(lambda: self.txt_live.setPlainText(""))
        self.btn_copy_live.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_live.toPlainText()))
        self.btn_export_live.clicked.connect(lambda: self.export_txt(self.txt_live.toPlainText()))

        row4.addWidget(self.btn_start_live)
        row4.addWidget(self.btn_stop_live)
        row4.addStretch(1)
        row4.addWidget(self.btn_clear_live)
        row4.addWidget(self.btn_copy_live)
        row4.addWidget(self.btn_export_live)
        box_l.addLayout(row4)

        layout.addWidget(box)

        self.lbl_status_live = QLabel("Prêt.")
        self.lbl_status_live.setWordWrap(True)
        layout.addWidget(self.lbl_status_live)

        self.txt_live = QTextEdit()
        self.txt_live.setReadOnly(False)
        layout.addWidget(self.txt_live, 1)

    @Slot()
    def start_live(self):
        if self.live_thread is not None:
            return

        try:
            label = self.combo_mic.currentText()
            input_device = self._mic_map.get(label, None)
            model_name = self.combo_model_live.currentText()
            language = self.combo_lang_live.currentData()
            beam = int(self.spin_beam_live.value())
            sr = int(self.spin_sr.value())
            chunk = float(self.spin_chunk.value())
            min_rms_v = float(self.spin_min_rms.value())
            compute_type = self.combo_compute_live.currentText()

            self.live_worker = LiveWorker(
                whisper=self.whisper,
                model_name=model_name,
                language=language,
                beam_size=beam,
                input_device=input_device,
                sample_rate=sr,
                chunk_seconds=chunk,
                min_rms=min_rms_v,
                compute_type=compute_type,
            )
            self.live_thread = QThread()
            self.live_worker.moveToThread(self.live_thread)

            self.live_thread.started.connect(self.live_worker.run)
            self.live_worker.textReady.connect(self._append_live_text)
            self.live_worker.status.connect(self._set_live_status)
            self.live_worker.error.connect(self._on_live_error)
            self.live_worker.stopped.connect(self._on_live_stopped)

            self.btn_start_live.setEnabled(False)
            self.btn_stop_live.setEnabled(True)
            self._set_live_status("Initialisation...")

            self.live_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            self._cleanup_live()

    @Slot()
    def stop_live(self):
        if self.live_worker is not None:
            self.live_worker.stop()
        self.btn_stop_live.setEnabled(False)
        self._set_live_status("Arrêt en cours...")

    @Slot(str)
    def _append_live_text(self, line: str):
        cur = self.txt_live.toPlainText()
        if cur and not cur.endswith("\n"):
            cur += "\n"
        cur += line
        self.txt_live.setPlainText(cur)
        self.txt_live.moveCursor(self.txt_live.textCursor().End)

    @Slot(str)
    def _set_live_status(self, s: str):
        self.lbl_status_live.setText(s)

    @Slot(str)
    def _on_live_error(self, msg: str):
        QMessageBox.critical(self, "Erreur Live", msg)

    @Slot()
    def _on_live_stopped(self):
        self._cleanup_live()
        self._set_live_status("Prêt.")

    def _cleanup_live(self):
        if self.live_thread is not None:
            self.live_thread.quit()
            self.live_thread.wait(2000)
        self.live_thread = None
        self.live_worker = None
        self.btn_start_live.setEnabled(True)
        self.btn_stop_live.setEnabled(False)

    # ---------- File tab ----------
    def _build_file_tab(self):
        layout = QVBoxLayout()
        self.tab_file.setLayout(layout)

        box = QGroupBox("Transcription Fichier")
        box_l = QVBoxLayout()
        box.setLayout(box_l)

        row1 = QHBoxLayout()
        self.edt_path = QLineEdit()
        self.edt_path.setReadOnly(True)
        self.btn_browse = QPushButton("Choisir un fichier...")
        self.btn_browse.clicked.connect(self.browse_file)
        row1.addWidget(self.edt_path, 1)
        row1.addWidget(self.btn_browse)
        box_l.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Langue:"))
        self.combo_lang_file = QComboBox()
        for label, code in self._language_choices():
            self.combo_lang_file.addItem(label, userData=code)
        row2.addWidget(self.combo_lang_file)

        row2.addWidget(QLabel("Modèle:"))
        self.combo_model_file = QComboBox()
        for m in self._model_choices():
            self.combo_model_file.addItem(m)
        self.combo_model_file.setCurrentText("medium")
        row2.addWidget(self.combo_model_file)

        row2.addWidget(QLabel("Compute:"))
        self.combo_compute_file = QComboBox()
        for c in self._compute_choices():
            self.combo_compute_file.addItem(c)
        self.combo_compute_file.setCurrentText("int8")
        row2.addWidget(self.combo_compute_file)

        row2.addWidget(QLabel("Beam:"))
        self.spin_beam_file = QSpinBox()
        self.spin_beam_file.setRange(1, 10)
        self.spin_beam_file.setValue(5)
        row2.addWidget(self.spin_beam_file)
        box_l.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_transcribe_file = QPushButton("Transcrire")
        self.btn_transcribe_file.clicked.connect(self.transcribe_file)
        self.btn_copy_file = QPushButton("Copier")
        self.btn_copy_file.clicked.connect(lambda: QApplication.clipboard().setText(self.txt_file.toPlainText()))
        self.btn_export_file = QPushButton("Exporter TXT")
        self.btn_export_file.clicked.connect(lambda: self.export_txt(self.txt_file.toPlainText()))
        self.btn_clear_file = QPushButton("Effacer")
        self.btn_clear_file.clicked.connect(lambda: self.txt_file.setPlainText(""))

        row3.addWidget(self.btn_transcribe_file)
        row3.addStretch(1)
        row3.addWidget(self.btn_clear_file)
        row3.addWidget(self.btn_copy_file)
        row3.addWidget(self.btn_export_file)
        box_l.addLayout(row3)

        layout.addWidget(box)

        self.lbl_status_file = QLabel("Prêt.")
        self.lbl_status_file.setWordWrap(True)
        layout.addWidget(self.lbl_status_file)

        self.txt_file = QTextEdit()
        layout.addWidget(self.txt_file, 1)

    @Slot()
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choisir un fichier audio",
            "",
            "Audio (*.*)",
        )
        if path:
            self.edt_path.setText(path)

    @Slot()
    def transcribe_file(self):
        if self.file_thread is not None:
            return

        src = self.edt_path.text().strip()
        if not src or not os.path.exists(src):
            QMessageBox.warning(self, "Fichier", "Sélectionne un fichier audio.")
            return

        try:
            model_name = self.combo_model_file.currentText()
            language = self.combo_lang_file.currentData()
            beam = int(self.spin_beam_file.value())
            compute_type = self.combo_compute_file.currentText()

            self.file_worker = FileWorker(
                whisper=self.whisper,
                src_path=src,
                model_name=model_name,
                language=language,
                beam_size=beam,
                compute_type=compute_type,
            )
            self.file_thread = QThread()
            self.file_worker.moveToThread(self.file_thread)

            self.file_thread.started.connect(self.file_worker.run)
            self.file_worker.textReady.connect(self._set_file_text)
            self.file_worker.status.connect(self._set_file_status)
            self.file_worker.error.connect(self._on_file_error)
            self.file_worker.finished.connect(self._on_file_finished)

            self.btn_transcribe_file.setEnabled(False)
            self._set_file_status("Initialisation...")
            self.file_thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Erreur", str(e))
            self._cleanup_file()

    @Slot(str)
    def _set_file_text(self, s: str):
        self.txt_file.setPlainText(s)

    @Slot(str)
    def _set_file_status(self, s: str):
        self.lbl_status_file.setText(s)

    @Slot(str)
    def _on_file_error(self, msg: str):
        QMessageBox.critical(self, "Erreur Fichier", msg)

    @Slot()
    def _on_file_finished(self):
        self._cleanup_file()
        self._set_file_status("Prêt.")

    def _cleanup_file(self):
        if self.file_thread is not None:
            self.file_thread.quit()
            self.file_thread.wait(2000)
        self.file_thread = None
        self.file_worker = None
        self.btn_transcribe_file.setEnabled(True)

    # ---------- Export ----------
    def export_txt(self, content: str):
        if not content.strip():
            QMessageBox.information(self, "Export", "Aucun texte à exporter.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter TXT", "transcription.txt", "TXT (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content.strip() + "\n")
        except Exception as e:
            QMessageBox.critical(self, "Export", str(e))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()