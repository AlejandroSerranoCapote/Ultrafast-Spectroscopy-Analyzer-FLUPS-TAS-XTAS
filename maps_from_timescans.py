import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QPushButton, QListWidget, QLineEdit, QLabel, 
                             QFileDialog, QMessageBox, QHBoxLayout, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt

BUTTON_STYLE = """
    QPushButton {
        background-color: #1e1e1e;
        color: #00bfff;
        border: 2px solid #00bfff;
        border-radius: 8px;
        padding: 4px 10px;
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;
    }
    QPushButton:hover { background-color: #00bfff; color: #000000; border: 2px solid #ffffff; }
    QPushButton:disabled { border: 2px solid #444; color: #666; background-color: #2a2a2a; }
"""

YELLOW_BUTTON_STYLE = BUTTON_STYLE.replace("#00bfff", "#f1c40f")
RESET_BUTTON_STYLE = BUTTON_STYLE.replace("#00bfff", "#ff4444")

DARK_THEME_STYLE = """
    QMainWindow, QWidget { background-color: #1e1e1e; color: #f0f0f0; font-family: "Segoe UI"; font-size: 8pt; }
    QGroupBox { border: 1px solid #00bfff; border-radius: 5px; margin-top: 12px; padding-top: 10px; font-weight: bold; color: #00bfff; }
    QLineEdit, QListWidget { background-color: #2d2d2d; border: 1px solid #555; border-radius: 4px; color: white; padding: 2px; }
    QLabel { font-weight: normal; }
"""

class XFELProcessor:
    def process(self, file_paths, energies, keys, time_scale=1.0):
        temp_d = []
        common_td = None
        
        for path in file_paths:
            data = np.load(path, allow_pickle=True).item()
            try:
                td = data[keys['time']] * time_scale
                
                if keys['direct_sig'].strip():
                    if keys['direct_sig'] in data:
                        sig = data[keys['direct_sig']]
                    else:
                        raise KeyError(f"La clave '{keys['direct_sig']}' no existe.")
                else:
                    sig = data[keys['es']] - data[keys['gs']]
                    
            except KeyError as e:
                raise KeyError(f"Error en {os.path.basename(path)}: {str(e)}")
            
            temp_d.append(sig)
            if common_td is None: common_td = td

        M = np.column_stack(temp_d)
        mask = ~np.isnan(common_td)
        return common_td[mask], np.array(energies), M[mask]

    def analyze_units(self, file_path, time_key):
        """Analiza estadísticamente el vector de tiempos del primer archivo."""
        try:
            data = np.load(file_path, allow_pickle=True).item()
            td = data[time_key]
            td = td[~np.isnan(td)]
            max_val = np.abs(td).max()
            step = np.mean(np.diff(np.sort(td)))
            
            if max_val < 50 and step < 0.5:
                return "ps (Picosegundos)", f"Máximo: {max_val:.2f}, Step medio: {step:.4f}"
            else:
                return "fs (Femtosegundos)", f"Máximo: {max_val:.1f}, Step medio: {step:.2f}"
        except Exception as e:
            return "Error", str(e)

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = XFELProcessor()
        self.file_list = []
        self.initUI()
        self.setStyleSheet(DARK_THEME_STYLE)

    def initUI(self):
        self.setWindowTitle("2D Maps from timescans builder")
        self.setGeometry(100, 100, 700, 850)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # --- CONFIGURACIÓN ---
        config_group = QGroupBox("MAPPING & SCALE CONFIGURATION (In .npy)")
        grid = QGridLayout()
        
        self.key_time = QLineEdit("Delay_fs_TT")
        self.key_es = QLineEdit("ES")
        self.key_gs = QLineEdit("GS")
        self.key_sig = QLineEdit("")
        self.key_sig.setPlaceholderText("Opcional: Diff, Intensity...") # Placeholder también aquí
        self.time_scale = QLineEdit("1.0") 
        
        grid.addWidget(QLabel("Time Key:"), 0, 0)
        grid.addWidget(self.key_time, 0, 1)
        grid.addWidget(QLabel("Time Scale Factor:"), 0, 2)
        grid.addWidget(self.time_scale, 0, 3)
        
        grid.addWidget(QLabel("Excited State Key (ES):"), 1, 0)
        grid.addWidget(self.key_es, 1, 1)
        grid.addWidget(QLabel("Ground State Key (GS):"), 1, 2)
        grid.addWidget(self.key_gs, 1, 3)
        
        grid.addWidget(QLabel("<b>Direct Signal Key:</b>"), 2, 0)
        grid.addWidget(self.key_sig, 2, 1, 1, 3)
        
        config_group.setLayout(grid)
        layout.addWidget(config_group)

        # --- ENERGÍAS ---
        layout.addWidget(QLabel("<b>ENERGY (eV) / WAVELENGTH (nm) VECTOR:</b>"))
        e_lay = QHBoxLayout()
        self.e_input = QLineEdit()
        
        self.e_input.setPlaceholderText("Ej: 2470.5, 2475.5, 2480.0 ...") 
        
        self.e_input.textChanged.connect(self.validate_counts)
        e_lay.addWidget(self.e_input)
        btn_e = QPushButton("IMPORT TXT")
        btn_e.setStyleSheet(BUTTON_STYLE)
        btn_e.clicked.connect(self.import_energies)
        e_lay.addWidget(btn_e)
        layout.addLayout(e_lay)

        # --- ARCHIVOS ---
        layout.addWidget(QLabel("<b>KINETIC DATA FILES (.npy):</b>"))
        f_lay = QHBoxLayout()
        btn_f = QPushButton("SELECT .NPY FILES")
        btn_f.setStyleSheet(BUTTON_STYLE)
        btn_f.clicked.connect(self.load_files)
        
        btn_check = QPushButton("CHECK UNITS")
        btn_check.setStyleSheet(YELLOW_BUTTON_STYLE)
        btn_check.clicked.connect(self.check_units)
        
        f_lay.addWidget(btn_f)
        f_lay.addWidget(btn_check)
        layout.addLayout(f_lay)
        
        self.list_w = QListWidget()
        layout.addWidget(self.list_w)

        self.label_status = QLabel("Ready")
        self.label_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label_status)

        # --- ACCIONES ---
        act_lay = QHBoxLayout()
        self.btn_run = QPushButton("GENERATE MAP")
        self.btn_run.setStyleSheet(BUTTON_STYLE)
        self.btn_run.clicked.connect(self.generate)
        
        self.btn_save = QPushButton("SAVE MAP")
        self.btn_save.setStyleSheet(BUTTON_STYLE)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)

        self.btn_reset = QPushButton("RESET")
        self.btn_reset.setStyleSheet(RESET_BUTTON_STYLE)
        self.btn_reset.clicked.connect(self.reset_app)

        act_lay.addWidget(self.btn_run)
        act_lay.addWidget(self.btn_save)
        act_lay.addWidget(self.btn_reset)
        layout.addLayout(act_lay)

    def check_units(self):
        if not self.file_list:
            QMessageBox.warning(self, "Error", "Carga archivos primero.")
            return
        unit, desc = self.processor.analyze_units(self.file_list[0], self.key_time.text())
        QMessageBox.information(self, "Unit Analysis", f"Detección: {unit}\n{desc}")

    def reset_app(self):
        self.file_list = []
        self.list_w.clear()
        self.e_input.clear()
        self.btn_save.setEnabled(False)
        self.validate_counts()

    def validate_counts(self):
        ne = len([x for x in self.e_input.text().split(',') if x.strip()])
        nf = len(self.file_list)
        if nf > 0 and nf == ne:
            self.label_status.setText(f"MATCH: {nf} Files")
            self.label_status.setStyleSheet("color: #00ff00; font-weight: bold;")
        else:
            self.label_status.setText(f"MISMATCH: {nf} Files / {ne} Energies")
            self.label_status.setStyleSheet("color: #ff4444; font-weight: bold;")

    def import_energies(self):
            path, _ = QFileDialog.getOpenFileName(self, "Load", "", "Text (*.txt *.csv *.dat)")
            if path:
                try:
                    with open(path, 'r') as f:
                        content = f.read()
                    
                    content = content.replace(',', ' ').replace('\n', ' ')
                    
                    d = np.fromstring(content, sep=' ')
                    
                    self.e_input.setText(", ".join(map(str, d)))
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"No se pudo importar el archivo:\n{e}")

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select", "", "Numpy (*.npy)")
        if files:
            self.file_list = sorted(files)
            self.list_w.clear()
            for f in self.file_list: self.list_w.addItem(os.path.basename(f))
            self.validate_counts()

    def generate(self):
        try:
            es = [float(x.strip()) for x in self.e_input.text().split(',') if x.strip()]
            ks = {'time': self.key_time.text(), 'es': self.key_es.text(), 
                  'gs': self.key_gs.text(), 'direct_sig': self.key_sig.text()}
            scale = float(self.time_scale.text())
            
            self.td, self.wl, self.m = self.processor.process(self.file_list, es, ks, scale)
            self.btn_save.setEnabled(True)
            
            plt.style.use('default') 
            
            plt.figure("XFEL 2D Map", figsize=(9, 7))
            plt.pcolormesh(self.wl, self.td, self.m, shading='auto', cmap='RdBu_r')
            plt.colorbar(label='Intensity')
            plt.xlabel('Energy / WL')
            
            unit_label = "ps" if scale == 0.001 else "fs"
            plt.ylabel(f'Delay ({unit_label})')
            
            plt.tight_layout()
            plt.show()
        except Exception as ex:
            QMessageBox.critical(self, "Processing Error", str(ex))

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "2D_Map_Export.npy", "Numpy (*.npy)")
        if path:
            np.save(path, {'data_c': self.m.T, 'WL': self.wl, 'TD': self.td})
            QMessageBox.information(self, "Done", "Saved successfully.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow(); ex.show()
    sys.exit(app.exec_())