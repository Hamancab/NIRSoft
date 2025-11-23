import sys
import os
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter, QFileDialog,
    QGroupBox, QFormLayout, QVBoxLayout, QPushButton,
    QLabel, QLineEdit, QDateEdit, QComboBox, QTextEdit,
    QListWidget, QScrollArea, QProgressBar, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QDate, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPalette, QColor
from PyQt5.Qt import QDesktopServices
import analysis  # full_analysys ve resource_path fonksiyonlarını içeren modül

class AnalysisThread(QThread):
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, analysis_args):
        super().__init__()
        self.analysis_args = analysis_args

    def run(self):
        try:
            pdf_path = analysis.full_analysys(**self.analysis_args)
            pngs = sorted(
                f for f in os.listdir(self.analysis_args['save_dir'])
                if f.lower().endswith('.png')
            )
            self.finished.emit(pdf_path, pngs)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NIRSoft')
        self.resize(1200, 800)
        self.nirs_path = ''
        self.save_dir  = ''
        self.pdf_path  = ''
        self.graph_map = {}
        self.trigger_indices = []
        self.trigger_times = []
        self._init_ui()

    def _init_ui(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet('QSplitter::handle { background: #3c3c3c }')

        # Left panel
        left_content = QWidget()
        left_content.setStyleSheet('background-color: #2d2d2d; color: white;')
        lyt = QVBoxLayout(left_content)
        lyt.setContentsMargins(8, 8, 8, 8)
        lyt.setSpacing(12)

        # Data group
        gb_data = QGroupBox('')
        gb_data.setStyleSheet(
            'QGroupBox { background: transparent; border: 1px solid #555; }'
        )
        gb_data.setMinimumWidth(300)
        dlyt = QVBoxLayout()
        dlyt.setContentsMargins(10, 10, 10, 10)
        dlyt.setSpacing(10)

        self.btn_open_nirs = QPushButton('📂 Select .nirs')
        dlyt.addWidget(self.btn_open_nirs)

        # Trigger selection
        tlyt = QFormLayout()
        # tlyt.setContentsMargins(10, 10, 10, 10)
        # tlyt.setSpacing(10)

        self.cb_start_trigger = QComboBox()
        self.cb_start_trigger.setPlaceholderText("Select Start Trigger")
        self.cb_start_trigger.addItem("First Trigger", -1)  # Default item
        self.cb_start_trigger.setCurrentIndex(0)  # Varsayılan olarak seçili
        self.cb_start_trigger.setStyleSheet("QComboBox { background-color: #2d2d2d; color: white; } QComboBox QAbstractItemView { background-color: #3c3c3c; color: white; }")


        self.cb_end_trigger = QComboBox()
        self.cb_end_trigger.setPlaceholderText("Select End Trigger")
        self.cb_end_trigger.addItem("Last Trigger", -1)  # Default item
        self.cb_end_trigger.setCurrentIndex(0)  # Varsayılan olarak seçili
        self.cb_end_trigger.setStyleSheet("QComboBox { background-color: #2d2d2d; color: white; } QComboBox QAbstractItemView { background-color: #3c3c3c; color: white; }")


        tlyt.addRow("Start Trigger:", self.cb_start_trigger)
        tlyt.addRow("End Trigger:", self.cb_end_trigger)

        dlyt.addLayout(tlyt)
        gb_data.setLayout(dlyt)
        lyt.addWidget(gb_data)

        # Patient info
        gb_info = QGroupBox('')
        gb_info.setStyleSheet(gb_data.styleSheet())
        gb_info.setMinimumWidth(300)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setContentsMargins(10, 10, 10, 10)
        self.setStyleSheet(
            self.styleSheet() +
            'QLabel { background: transparent; border: none; color: white; }'
        )

        self.le_name  = QLineEdit(); self.le_name.setPlaceholderText('Patient Name')
        self.le_id    = QLineEdit(); self.le_id.setPlaceholderText('Patient ID')
        self.de_dob   = QDateEdit(calendarPopup=True)
        self.de_dob.setDisplayFormat('dd-MM-yyyy')
        self.de_dob.setDate(QDate.currentDate().addYears(-30))
        self.de_anal  = QDateEdit(calendarPopup=True)
        self.de_anal.setDisplayFormat('dd-MM-yyyy')
        self.de_anal.setDate(QDate.currentDate())
        self.le_proto = QLineEdit(); self.le_proto.setPlaceholderText('Protocol Name')
        self.le_diag  = QLineEdit(); self.le_diag.setPlaceholderText('Diagnosis')
        self.te_notes = QTextEdit(); self.te_notes.setPlaceholderText('Notes')
        self.te_notes.setMaximumHeight(100)

        # Gender selection
        self.cb_gender = QComboBox()
        self.cb_gender.addItems(['Male', 'Female', 'Other'])
        self.cb_gender.setCurrentIndex(2)  # Varsayılan olarak 'Other' seçili
        self.cb_gender.setStyleSheet("QComboBox { background-color: #2d2d2d; color: white; } QComboBox QAbstractItemView { background-color: #3c3c3c; color: white; }")

        for w in (self.le_name, self.le_id, self.le_proto, self.le_diag):
            w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        form.addRow('Name:',        self.le_name)
        form.addRow('Patient ID:',  self.le_id)
        form.addRow('Birth Date:',  self.de_dob)
        form.addRow('Analysis Dt:', self.de_anal)
        form.addRow('Gender:',      self.cb_gender)
        form.addRow('Protocol:',    self.le_proto)
        form.addRow('Diagnosis:',   self.le_diag)
        form.addRow('Notes:',       self.te_notes)

        gb_info.setLayout(form)
        lyt.addWidget(gb_info)

        # Controls
        gb_ctrl = QGroupBox('')
        gb_ctrl.setStyleSheet(gb_data.styleSheet())
        gb_ctrl.setMinimumWidth(300)
        cyl = QVBoxLayout()
        cyl.setContentsMargins(10, 10, 10, 10)
        cyl.setSpacing(10)
        self.btn_run  = QPushButton('Run Analysis')
        self.btn_run.setStyleSheet(
            'background-color: #d32f2f; color: white; font-weight: bold;'
        )
        self.progress = QProgressBar(); self.progress.setRange(0, 0); self.progress.hide()
        self.btn_pdf  = QPushButton('Open PDF'); self.btn_pdf.setEnabled(False)
        cyl.addWidget(self.btn_run)
        cyl.addWidget(self.progress)
        cyl.addWidget(self.btn_pdf)
        gb_ctrl.setLayout(cyl)
        lyt.addWidget(gb_ctrl)

        # Graphs list
        gb_list = QGroupBox('')
        gb_list.setStyleSheet(gb_data.styleSheet())
        gb_list.setMinimumWidth(300)
        vlist = QVBoxLayout()
        vlist.setContentsMargins(10, 10, 10, 10)
        vlist.setSpacing(10)
        self.lst_graphs = QListWidget()
        self.lst_graphs = QListWidget()
        self.lst_graphs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        vlist.addWidget(self.lst_graphs)
        gb_list.setLayout(vlist)
        lyt.addWidget(gb_list, 1)
        lyt.addStretch()

        # Wrap left panel
        scroll_left = QScrollArea()
        scroll_left.setWidgetResizable(True)
        scroll_left.setWidget(left_content)
        scroll_left.setMinimumWidth(320)
        splitter.addWidget(scroll_left)

        # Right panel
        self.lbl_display = QLabel(alignment=Qt.AlignCenter)
        self.lbl_display.setStyleSheet('background-color: #1e1e1e;')
        scroll_right = QScrollArea()
        scroll_right.setWidgetResizable(True)
        scroll_right.setWidget(self.lbl_display)
        splitter.addWidget(scroll_right)
        splitter.setSizes([350, 850])

        self.setCentralWidget(splitter)

        # Connections
        self.btn_open_nirs.clicked.connect(self.open_nirs)
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_pdf.clicked.connect(self.open_pdf)
        self.lst_graphs.currentTextChanged.connect(self.display_graph)

    def open_nirs(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, 'Select .nirs File', filter='NIRS Files (*.nirs)'
        )
        if fn:
            self.nirs_path = fn
            self.load_triggers()

    def load_triggers(self):
        try:
            self.trigger_indices, self.trigger_times = analysis.get_trigger_indices(self.nirs_path)
            self.cb_start_trigger.clear()
            self.cb_end_trigger.clear()
            for i, t in enumerate(self.trigger_times):
                label = f'Trigger{i+1}: {t:.2f} s'
                self.cb_start_trigger.addItem(label, i)
                self.cb_end_trigger.addItem(label, i)

            # Otomatik seçim: Start Trigger = Trigger 1, End Trigger = Son Trigger
            self.cb_start_trigger.setCurrentIndex(0)
            self.cb_end_trigger.setCurrentIndex(len(self.trigger_times) - 1)
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to load triggers: {e}')

    def run_analysis(self):
        if not self.nirs_path:
            QMessageBox.warning(self, 'Missing File', 'Please select a .nirs file first.')
            return
        start = self.cb_start_trigger.currentData()
        end   = self.cb_end_trigger.currentData()
        if start is None or end is None or start >= end:
            QMessageBox.warning(
                self, 'Invalid Selection', 'Please select valid start and end triggers.'
            )
            return
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet('background-color: grey;')
        self.progress.show()
        QApplication.processEvents()

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(self.save_dir, exist_ok=True)

        probe_mat = analysis.resource_path('probe.mat')
        def get_text(w): return w.text().strip() if hasattr(w, 'text') else ''
        def get_plain(w): return w.toPlainText().strip() if hasattr(w, 'toPlainText') else ''
        def get_date(w): return w.date().toString('dd-MM-yyyy') if hasattr(w, 'date') else ''
        def get_combo(w): return w.currentText().strip() if hasattr(w, 'currentText') else ''

        args = {
            'nirs_file_path': self.nirs_path,
            'hasta_adi':      get_text(self.le_name),
            'patient_id':     get_text(self.le_id),
            'date_of_birth':  get_date(self.de_dob),
            'analysis_date':  get_date(self.de_anal),
            'gender':         get_combo(self.cb_gender),
            'protocol_name':  get_text(self.le_proto),
            'diagnosis':      get_text(self.le_diag),
            'notes':          get_plain(self.te_notes),
            'save_dir':       self.save_dir,
            'probe_mat_path': probe_mat,
            'start_trigger':  start,
            'end_trigger':    end
        }
        self.thread = AnalysisThread(args)
        self.thread.finished.connect(self.on_analysis_finished)
        self.thread.error.connect(self.on_analysis_error)
        self.thread.start()

    def on_analysis_finished(self, pdf_path, pngs):
        self.pdf_path = pdf_path
        self.lst_graphs.clear()
        self.graph_map.clear()
        for f in pngs:
            self.lst_graphs.addItem(f)
            self.graph_map[f] = os.path.join(self.save_dir, f)
        if pngs:
            self.lst_graphs.setCurrentRow(0)
        self.btn_pdf.setEnabled(True)
        QMessageBox.information(
            self, 'Done', f'Analysis complete!\nPDF at:\n{pdf_path}'
        )
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_run.setStyleSheet(
            'background-color: #d32f2f; color: white;'
        )

    def on_analysis_error(self, msg):
        QMessageBox.critical(self, 'Analysis Error', msg)
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_run.setStyleSheet(
            'background-color: #d32f2f; color: white;'
        )

    def display_graph(self, name):
        path = self.graph_map.get(name)
        if path and os.path.exists(path):
            pix = QPixmap(path)
            self.lbl_display.setPixmap(
                pix.scaled(
                    self.lbl_display.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

    def open_pdf(self):
        if self.pdf_path and os.path.exists(self.pdf_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.pdf_path))
        else:
            QMessageBox.warning(self, 'No PDF', 'PDF not found.')


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor('#2d2d2d'))
    pal.setColor(QPalette.WindowText, Qt.white)
    pal.setColor(QPalette.Base, QColor('#1e1e1e'))
    pal.setColor(QPalette.AlternateBase, QColor('#2d2d2d'))
    pal.setColor(QPalette.ToolTipBase, Qt.white)
    pal.setColor(QPalette.ToolTipText, Qt.white)
    pal.setColor(QPalette.Text, Qt.white)
    pal.setColor(QPalette.Button, QColor('#3c3c3c'))
    pal.setColor(QPalette.ButtonText, Qt.white)
    pal.setColor(QPalette.Highlight, QColor('#2a82da'))
    pal.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(pal)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
