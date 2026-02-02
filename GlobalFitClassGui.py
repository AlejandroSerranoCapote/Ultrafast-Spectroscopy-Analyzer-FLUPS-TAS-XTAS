import os
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QMessageBox, QProgressBar, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox, 
    QFormLayout, QWidget, QTabWidget, QApplication, QInputDialog,
    QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Import your physics modules
import fit
# import core_analysis # Uncomment if strictly needed, usually accessed via fit or parent

# --- 1. ESTILO NEÓN COMPACTO ---
BUTTON_STYLE = """
    QPushButton {
        background-color: #1e1e1e;
        color: #00bfff;
        border: 2px solid #00bfff;
        border-radius: 8px;          /* Menos redondeado */
        padding: 4px 10px;           /* MUCHO MENOS Relleno (antes 8px 16px) */
        font-weight: bold;
        font-family: "Segoe UI";
        font-size: 9pt;              /* Fuente más pequeña (antes 11pt) */
    }
    QPushButton:hover {
        background-color: #00bfff;
        color: #000000;
        border: 2px solid #ffffff;
    }
    QPushButton:pressed {
        background-color: #008cb3;
        border: 2px solid #008cb3;
        color: white;
        padding-top: 6px;
        padding-left: 12px;
    }
    QPushButton:disabled {
        border: 2px solid #444;
        color: #666;
        background-color: #2a2a2a;
    }
"""

# --- 2. TEMA OSCURO GENERAL COMPACTO ---
DARK_THEME_STYLE = """
    QDialog, QWidget {
        background-color: #1e1e1e;
        color: #f0f0f0;
        font-family: "Segoe UI";
        font-size: 8pt;              /* Fuente general más pequeña (antes 10pt) */
    }
    QGroupBox {
        border: 1px solid #00bfff;
        border-radius: 5px;
        margin-top: 8px;             /* Menos margen arriba */
        padding-top: 10px;
        font-weight: bold;
        color: #00bfff;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 3px;
    }
    QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
        background-color: #2d2d2d;
        border: 1px solid #555;
        border-radius: 4px;
        color: white;
        padding: 2px;                /* Menos relleno interno */
        min-height: 18px;            /* Altura mínima forzada pequeña */
    }
    QSpinBox:focus, QComboBox:focus {
        border: 1px solid #00bfff;
    }
    QComboBox QAbstractItemView {
        background-color: #2d2d2d;
        color: white;
        selection-background-color: #00bfff;
        selection-color: black;
    }
    QLabel { color: #f0f0f0; }
    QCheckBox { color: #f0f0f0; }
    
    QProgressBar {
        border: 1px solid #555;
        border-radius: 5px;
        text-align: center;
        color: white;
        max-height: 15px;            /* Barra de progreso más fina */
    }
    QProgressBar::chunk {
        background-color: #00bfff;
        width: 10px;
        margin: 0.5px;
    }
"""



class Surface3DWindow(QDialog):
    """Ventana independiente para visualizar el 3D sin bloquear el panel principal."""
    def __init__(self, xs, ys, zs, scale='linear', parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Surface Preview")
        self.resize(800, 600)
        
        self.setStyleSheet(DARK_THEME_STYLE)

        self.setWindowModality(Qt.NonModal)

        layout = QVBoxLayout()
        
        self.fig = plt.Figure(facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.fig)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.toolbar.setStyleSheet("QToolBar { background-color: transparent; border: none; }")
        
        self._make_icons_white(self.toolbar)
        
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.plot_data(xs, ys, zs, scale)

    def _make_icons_white(self, toolbar):
        """Recorre los botones de la toolbar y pinta los iconos de blanco."""
        for action in toolbar.actions():
            icon = action.icon()
            if icon.isNull(): continue
            
            pixmap = icon.pixmap(32, 32)
            
            if not pixmap.isNull():
                mask = pixmap.mask()
                pixmap.fill(QColor("white")) # Rellenar todo de blanco
                pixmap.setMask(mask)         # Recortar con la forma original
                action.setIcon(QIcon(pixmap))

    def plot_data(self, xs, ys, zs, scale):
        ax = self.fig.add_subplot(111, projection='3d')
        
        # Colores para modo oscuro
        ax.set_facecolor('#1e1e1e')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='z', colors='white')

        X, Y = np.meshgrid(xs, ys)
        Z = zs.T
        
        surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none', antialiased=True)
        
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Delay (ps)")
        ax.set_zlabel("Transient absorption")
        
        # Limpiar paneles
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.view_init(elev=25, azim=75)
        
        if scale == 'symlog':
            ax.set_yscale('symlog', linthresh=1.0)
            
        cbar = self.fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        self.canvas.draw()
        
class GlobalFitPanel(QDialog):
    def __init__(self, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Global Fit Analysis")
            
            screen = QApplication.primaryScreen()
            screen_geom = screen.availableGeometry() # Tamaño útil (sin barra tareas)
            
            # Calculamos el 85% del ancho y alto
            w_target = int(screen_geom.width() * 0.8)
            h_target = int(screen_geom.height() * 0.65)
            
            x_pos = (screen_geom.width() - w_target) // 2 + screen_geom.left()
        
            y_pos = screen_geom.top() + 50
            
            # Aplicamos posición (x, y) y tamaño (w, h)
            self.setGeometry(x_pos, y_pos, w_target, h_target)
            self.setStyleSheet(DARK_THEME_STYLE + BUTTON_STYLE) # Apply Dark Theme

            # --- 1. Variables de Datos ---
           
            self.parent_app = parent
            self.data_c = None   
            self.data_raw = None 
            self.TD = None       
            self.WL = None       
            self.base_dir = None

            if hasattr(parent, "save_dir") and parent.save_dir:
                self.base_dir = parent.save_dir
            elif hasattr(parent, "file_path") and parent.file_path:
                base_name = os.path.splitext(os.path.basename(parent.file_path))[0]
                self.base_dir = os.path.join(os.path.dirname(parent.file_path), f"{base_name}_Results")
                os.makedirs(self.base_dir, exist_ok=True)
            else:
                self.base_dir = os.getcwd()
    
            # --- 2. Variables del Ajuste ---
            self.numExp = 2
            self.model_type = 'Parallel' 
            self.t0_choice = 'No'
            self.tech = 'TAS'
            self.yscale = 'linear'
            
            # Placeholders para resultados
            self.fit_result = None
            self.fit_x = None
            self.As = None
            # ... resto de variables fit
            self.fit_resid = None
            self.fit_fitres = None
            self.ci = None
            self.errAs = None
            self.t0s = None
            self.errt0s = None
            self.errtaus = None
            self.ini = None
            self.limi = None
            self.lims = None
    
            # --- 3. DISEÑO PRINCIPAL (LAYOUT) ---
            main_layout = QHBoxLayout() 
            
            # --- A. Panel Izquierdo (Sidebar) ---
            self.sidebar = QWidget()
            self.sidebar.setFixedWidth(340) 
            self.sidebar_layout = QVBoxLayout(self.sidebar)
            self.sidebar_layout.setContentsMargins(5, 5, 5, 5)
            
            self._init_sidebar_ui() 
            
            main_layout.addWidget(self.sidebar)
    
            # --- B. Panel Derecho (Gráficos) ---
            self.right_area = QWidget()
            self.right_layout = QVBoxLayout(self.right_area)
            
            self._init_plots_ui() 
            
            main_layout.addWidget(self.right_area)
    
            self.setLayout(main_layout)
    
            # --- IMPORTANTE: INICIALIZAR VARIABLES DE PLOTTING ---
            self.pcm_exp = None
            self.cbar_exp = None
            self.pcm_fit = None
            self.cbar_fit = None
            self.pcm_resid = None
            self.cbar_resid = None
        

    def _init_sidebar_ui(self):
        """Construye todos los botones y cajas del panel izquierdo."""
        l = self.sidebar_layout
        
        # --- Grupo 1: Carga de Datos ---
        gb_load = QGroupBox("1. Data Source")
        v_load = QVBoxLayout()
        
        self.label_status = QLabel("No data loaded")
        self.label_status.setStyleSheet("color: gray; font-style: italic; font-weight: bold;")
        v_load.addWidget(self.label_status)
        
        h_btns = QHBoxLayout()
        self.btn_load = QPushButton("Load .npy")
        self.btn_load.clicked.connect(self.load_data) 
        h_btns.addWidget(self.btn_load)
        
        self.btn_parent = QPushButton("Use Parent Data")
        self.btn_parent.clicked.connect(self.use_parent_data) 
        h_btns.addWidget(self.btn_parent)
        
        v_load.addLayout(h_btns)
        gb_load.setLayout(v_load)
        l.addWidget(gb_load)

        # --- Grupo 2: Pre-procesado ---
        gb_prep = QGroupBox("2. Pre-processing")
        form_prep = QFormLayout()

        # Baseline
        self.spin_bl = QSpinBox()
        self.spin_bl.setRange(0, 500)
        self.spin_bl.setValue(5)
        self.spin_bl.valueChanged.connect(self.apply_baseline_correction) 
        form_prep.addRow("Baseline Pts:", self.spin_bl)

        # Rangos WL
        self.spin_wl_min = QDoubleSpinBox(); self.spin_wl_min.setRange(0, 10000); 
        self.spin_wl_max = QDoubleSpinBox(); self.spin_wl_max.setRange(0, 10000); 
        self.spin_wl_max.setDecimals(6)    
        self.spin_wl_max.setSingleStep(0.5)
        self.spin_wl_min.setDecimals(6)
        self.spin_wl_min.setSingleStep(0.1)
        
        form_prep.addRow("Min WL (nm):", self.spin_wl_min)
        form_prep.addRow("Max WL (nm):", self.spin_wl_max)

        # Rangos Tiempo
        self.spin_t_min = QDoubleSpinBox(); self.spin_t_min.setRange(-100, 1e6); self.spin_t_min.setDecimals(3)
        self.spin_t_max = QDoubleSpinBox(); self.spin_t_max.setRange(-100, 1e6); self.spin_t_max.setDecimals(3)
        form_prep.addRow("Min Time (ps):", self.spin_t_min)
        form_prep.addRow("Max Time (ps):", self.spin_t_max)
        

        # Binning
        self.spin_bin = QSpinBox()
        self.spin_bin.setRange(1, 50)
        self.spin_bin.setValue(1)
        form_prep.addRow("Binning:", self.spin_bin)
        
        # Botón Preview
        self.btn_preview = QPushButton("Apply & Preview")
        self.btn_preview.clicked.connect(self._preview_data_processing) 
        form_prep.addRow(self.btn_preview)

        gb_prep.setLayout(form_prep)
        l.addWidget(gb_prep)

        # --- Grupo 3: Modelo ---
        gb_model = QGroupBox("3. Model Settings")
        form_model = QFormLayout()
        
        self.btn_svd = QPushButton("Run SVD Analysis")
        self.btn_svd.clicked.connect(self.run_svd)
        form_model.addRow(self.btn_svd)
        
        gb_vis = QGroupBox("4. Visualization")
        form_vis = QFormLayout()
        
        self.btn_plot_3d = QPushButton("Ver Mapa en 3D")
        self.btn_plot_3d.clicked.connect(self.plot_3d_surface)
        form_vis.addRow(self.btn_plot_3d)
        
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Linear", "SymLog"])
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed) # Conectamos función
        form_vis.addRow("Time Axis Scale:", self.combo_scale)
        
        gb_vis.setLayout(form_vis)
        l.addWidget(gb_vis)
        # Num Exponenciales
        self.spin_numExp = QSpinBox()
        self.spin_numExp.setRange(1, 6)
        self.spin_numExp.setValue(2)
        form_model.addRow("Exponentials:", self.spin_numExp)

        # Tipo de Modelo
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Parallel (DAS)", "Sequential (SAS)","Dumped Oscillation"])
        form_model.addRow("Model Type:", self.combo_model)

        # Técnica
        self.combo_tech = QComboBox()
        self.combo_tech.addItems(["FLUPS", "TAS", "TCSPC"])
        form_model.addRow("Technique:", self.combo_tech)

        # Chirp
        self.chk_chirp = QCheckBox("Fit Independent t0 (Chirp)")
        form_model.addRow(self.chk_chirp)
        
        # Initial Guesses
        self.btn_edit_guess = QPushButton("Edit Initial Guesses")
        self.btn_edit_guess.clicked.connect(self._open_guess_editor_and_update)
        form_model.addRow(self.btn_edit_guess)
        gb_model.setLayout(form_model)
        l.addWidget(gb_model)

        # --- Botones Finales ---
        self.btn_run = QPushButton("RUN FIT")
        self.btn_run.setFixedHeight(40)  
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.run_fit_pipeline) 
        l.addWidget(self.btn_run)
        
        self.btn_show_das = QPushButton("Show Plots / Results")
        self.btn_show_das.setEnabled(False)
        self.btn_show_das.clicked.connect(self.plot_das_and_more) 
        l.addWidget(self.btn_show_das)

        l.addStretch()
        
    def run_svd(self):
        if self.data_c is None:
            QMessageBox.warning(self, "Error", "Carga y procesa datos primero (Apply & Preview).")
            return
    
        # 1. Ejecutar SVD matemático
        # data_c suele ser [WL x TD]
        try:
            # Usamos economy SVD (compute_uv=True por defecto)
            U, s, Vh = np.linalg.svd(self.data_c, full_matrices=False)
            
            self.svd_U = U    # Vectores espectrales (Especies)
            self.svd_s = s    # Importancia de cada uno
            self.svd_V = Vh.T # Vectores temporales (Cinéticas)
    
            self._plot_svd_results()
            self.tabs.setCurrentWidget(self.tab_svd) # Cambiar a la pestaña SVD automáticamente
            
        except Exception as e:
            print(f"SVD Error: {e}")
    def _create_svd_canvas(self, tab_widget):
        fig = plt.Figure(figsize=(5, 8))
        # ax1: Scree Plot, ax2: Primeros Componentes Espectrales
        ax1 = fig.add_subplot(211) 
        ax2 = fig.add_subplot(212)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        return canvas, (ax1, ax2)

    def _plot_svd_results(self):
        ax1, ax2 = self.ax_svd
        ax1.clear()
        ax2.clear()
    
        # --- Plot 1: Scree Plot (Log scale) ---
        n_comp = min(len(self.svd_s), 10) # Ver top 10
        ax1.semilogy(range(1, n_comp + 1), self.svd_s[:n_comp], 'o-', color='red')
        ax1.set_title("Singular Values (Scree Plot)")
        ax1.set_ylabel("Eigenvalue (log)")
        ax1.set_xlabel("Component Number")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
    
        # 2. Componentes Espectrales 
        wl = getattr(self, '_wl_proc', self.WL)
        # Leemos el valor del SpinBox de la interfaz
        n_mostrar = self.spin_numExp.value() 
        
        for i in range(min(n_mostrar, len(self.svd_s))):
            ax2.plot(wl, self.svd_U[:, i], label=f"Comp {i+1}")
        
        ax2.set_title(f"First {n_mostrar} Spectral Components")
        ax2.set_xlabel("Energy / Wavelength")
        ax2.axhline(0, color='black', lw=1, alpha=0.5)
        ax2.legend()
        
        self.canvas_svd.draw()        
  
    def _on_scale_changed(self, text):
            """Actualiza la variable de escala y repinta los gráficos."""
            self.yscale = text.lower() # 'linear' o 'symlog'
            
            # Repintar todo lo que esté activo
            self._update_exp_canvas()
            self._update_fit_canvas()
            self._update_resid_canvas()

    def _init_plots_ui(self):
            """Construye los Tabs y gráficos del panel derecho."""
            l = self.right_layout
            
            # Tabs
            self.tabs = QTabWidget()
            
            self.tabs.setStyleSheet("""
                        QTabWidget::pane { 
                            border: 1px solid #999; 
                            background: white; 
                        }
                        QTabBar::tab { 
                            background: #e0e0e0; 
                            color: black;
                            padding: 8px 20px; 
                            border: 1px solid #bbb; 
                            border-bottom: none; 
                            border-top-left-radius: 4px; 
                            border-top-right-radius: 4px; 
                            margin-right: 2px;
                        }
                        QTabBar::tab:selected { 
                            background: #ffffff; 
                            /* font-weight: bold;  <--- LINEA BORRADA */
                            border-bottom: 1px solid #ffffff; 
                        }
                        QTabBar::tab:hover {
                            background: #d0d0d0;
                        }
                    """)
            
            self.tab_exp = QWidget()
            self.tab_fit = QWidget()
            self.tab_resid = QWidget()
            self.tab_svd = QWidget() 
            
            self.tabs.addTab(self.tab_exp, "Experimental")
            self.tabs.addTab(self.tab_fit, "Fit Reconstructed")
            self.tabs.addTab(self.tab_resid, "Residuals")
            self.tabs.addTab(self.tab_svd, "SVD Diagnosis")
            
            # Crear Canvas (usando helper)
            self.canvas_exp, self.ax_exp = self._create_canvas_for_tab(self.tab_exp)
            self.canvas_fit, self.ax_fit = self._create_canvas_for_tab(self.tab_fit)
            self.canvas_resid, self.ax_resid = self._create_canvas_for_tab(self.tab_resid)
            self.canvas_svd, self.ax_svd = self._create_svd_canvas(self.tab_svd)
            
            l.addWidget(self.tabs)
            
            # Barra de progreso
            self.progress_bar = QProgressBar()
            self.progress_bar.setValue(0)
            self.progress_bar.setTextVisible(True)
            l.addWidget(self.progress_bar)
    def plot_3d_surface(self):
        """Lanza la ventana 3D independiente (No Modal)."""
        if self.data_c is None:
            QMessageBox.warning(self, "Sin datos", "Aplica 'Preview' antes de ver el 3D.")
            return
    
        # Obtener datos actuales
        xs = getattr(self, '_wl_proc', self.WL)
        ys = getattr(self, '_td_proc', self.TD)
        zs = self.data_c
        scale = getattr(self, 'yscale', 'linear')
    
        # Crear y mostrar la ventana
        self.pop_3d = Surface3DWindow(xs, ys, zs, scale, parent=self)
        self.pop_3d.show() # .show() no bloquea la ejecución    
    def _generate_defaults(self):
            """Genera los valores iniciales (Guesses) basados en la configuración actual."""
            # 1. Leer configuración actual
            numExp = self.spin_numExp.value()
            t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
            tech = self.combo_tech.currentText()
            
            model_str = self.combo_model.currentText()
            is_oscillation = "Oscillation" in model_str
            
            if self.data_c is not None:
                numWL = self.data_c.shape[0]
            elif self.WL is not None:
                numWL = len(self.WL)
            else:
                QMessageBox.warning(self, "Warning", "Load data first to generate guesses.")
                return False
    
            # 2. Calcular tamaño vector L
            if is_oscillation:
                # Estructura: [w, t0, tau_1..n, alpha, omega, phi, (A1..An, B)_wl1, (A1..An, B)_wl2...]
                # Globales: w(1) + t0(1) + taus(numExp) + alpha(1) + omega(1) + phi(1)
                # Locales por WL: numExp amplitudes + 1 amplitud de oscilación (B)
                L = (2 + numExp + 3) + numWL * (numExp + 1)
            elif t0_choice == 'Yes':
                L = 1 + numExp + numWL*(numExp+1)
            else:
                L = 2 + numExp + numWL*numExp
                
            self.ini = np.zeros(L)
            self.limi = -np.inf * np.ones(L)
            self.lims = np.inf * np.ones(L)
    
            # 3. Rellenar valores
            taus_defaults = [0.5, 5.0, 50.0, 500.0, 2000.0, 5000.0]
            w_guess = 0.15 if tech == 'TAS' else (0.3 if tech == 'FLUPS' else 0.1)
            
            if is_oscillation:
                # Globales base
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0  # w
                self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0  # t0
                
                # Taus globales
                base_tau = 2
                for n in range(numExp):
                    val_t = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                    self.ini[base_tau + n] = val_t
                    self.limi[base_tau + n] = 0.001
                    self.lims[base_tau + n] = 1e8
                    
                # Parámetros de la oscilación (Globales)
                idx_osc = base_tau + numExp
                self.ini[idx_osc]   = 0.1   # alpha (amortiguamiento)
                self.ini[idx_osc+1] = 1.0   # omega (frecuencia angular)
                self.ini[idx_osc+2] = 0.0   # phi (fase)
                
                self.limi[idx_osc] = 0.0;     self.lims[idx_osc] = 100.0   # alpha bounds
                self.limi[idx_osc+1] = 0.0;   self.lims[idx_osc+1] = 500.0 # omega bounds
                self.limi[idx_osc+2] = -np.pi; self.lims[idx_osc+2] = np.pi # phi bounds
    
                # Amplitudes locales (Exponenciales + B)
                start_local = idx_osc + 3
                params_per_wl = numExp + 1 # A1..An + B
                val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                
                self.ini[start_local:] = val_A 
                # Opcional: inicializar B específicamente
                # self.ini[start_local + numExp :: params_per_wl] = val_A * 0.5
    
            elif t0_choice == 'No':
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                self.ini[1] = 0.0;     self.limi[1] = -5.0; self.lims[1] = 5.0
                
                base_tau = 2
                for n in range(numExp):
                    idx = base_tau + n
                    val_t = taus_defaults[n] if n < len(taus_defaults) else 1000.0*(n+1)
                    self.ini[idx] = val_t; self.limi[idx] = 0.001; self.lims[idx] = 1e8
                
                start_A = base_tau + numExp
                val_A = 1000.0 if tech == 'TCSPC' else (5.0 if tech == 'FLUPS' else 0.01)
                self.ini[start_A:] = val_A
                
            else:
                self.ini[0] = w_guess; self.limi[0] = 0.05; self.lims[0] = 2.0
                for n in range(numExp):
                    self.ini[1+n] = taus_defaults[n] if n < len(taus_defaults) else 100.0
                    self.limi[1+n] = 0.001; self.lims[1+n] = 1e8
                
                base_idx = 1 + numExp
                params_per_wl = 1 + numExp
                val_A = 1000.0 if tech == 'TCSPC' else 0.1
                self.ini[base_idx:] = val_A
                self.ini[base_idx::params_per_wl] = 0.0
                self.limi[base_idx::params_per_wl] = -5.0
                self.lims[base_idx::params_per_wl] = 5.0
                
            return True
    def _create_canvas_for_tab(self, tab_widget):
        """Helper para inicializar matplotlib dentro de un tab."""
        fig = plt.Figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        tab_widget.setLayout(layout)
        
        return canvas, ax

    # --- Métodos auxiliares de limpieza de UI ---
    def _clear_colorbar_if_exists(self, cbar):
        try:
            if cbar is not None: cbar.remove()
        except: pass

    def use_parent_data(self):
        """Cargar datos del parent y actualizar canvas"""
        self.update_from_parent()
        self.btn_run.setEnabled(True)
        self.btn_show_das.setEnabled(False)
        
    def update_from_parent(self):
         p = self.parent_app
         if p is None: return
             
         if getattr(p, "is_TAS_mode", False):
              if hasattr(p, "data_corrected") and p.data_corrected is not None:
                  incoming_data = np.array(p.data_corrected, copy=True)
         
         # Guardamos en RAW
         self.data_raw = incoming_data 
         self.WL = getattr(p, "WL", None)
         self.TD = getattr(p, "TD", None)
         
         self.apply_baseline_correction()
    def apply_baseline_correction(self):
            """Recalcula data_c desde data_raw usando el valor actual del SpinBox y actualiza el plot."""
            if self.data_raw is None:
                return
    
            n_pts = self.spin_bl.value()
            
            temp_data = self.data_raw.copy()
            
            if n_pts > 0:
                if temp_data.shape[1] >= n_pts:
                    # Calcular baseline (media de las primeras n columnas de tiempo)
                    # Asumiendo shape [NumWL, NumTD] o viceversa. 
                    baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
                    temp_data = temp_data - baseline
                else:
                    print("Warning: Not enough points for baseline.")
    
            # Actualizamos la variable oficial que usa el ajuste
            self.data_c = temp_data
            
            # Repintamos el canvas experimental inmediatamente
            self._update_exp_canvas()
# --- LÓGICA DE CARGA DE DATOS ---

    def _update_ui_limits_from_data(self):
        """Actualiza los rangos de las cajas numéricas (SpinBoxes) según los datos cargados."""
        if self.WL is not None and len(self.WL) > 0:
            self.spin_wl_min.setValue(np.min(self.WL))
            self.spin_wl_max.setValue(np.max(self.WL))
        
        if self.TD is not None and len(self.TD) > 0:
            self.spin_t_min.setValue(np.min(self.TD))
            self.spin_t_max.setValue(np.max(self.TD))
        
        # Al cargar, reseteamos data_c a raw y pintamos
        self.data_c = self.data_raw.copy()
        
        # Pintamos inmediatamente la data cruda
        self._update_exp_canvas(use_processed=False)

    def use_parent_data(self):
        """Cargar datos desde la ventana principal (si existe)."""
        if self.parent_app is None: return
        
        if hasattr(self.parent_app, "data_corrected") and self.parent_app.data_corrected is not None:
            self.data_raw = np.array(self.parent_app.data_corrected, copy=True)
            self.WL = getattr(self.parent_app, "WL", None)
            self.TD = getattr(self.parent_app, "TD", None)
            
            # Detectar técnica
            if getattr(self.parent_app, "is_TAS_mode", False):
                self.combo_tech.setCurrentText("TAS")
            else:
                self.combo_tech.setCurrentText("FLUPS")
                
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.label_status.setText(f"Loaded from Parent: {len(self.WL)} WL, {len(self.TD)} TD")

    def load_data(self):
        """Cargar desde .npy usando tu módulo 'fit'."""
        try:
            raw_data, TD, WL, base_dir = fit.load_npy(self)
            
            self.data_raw = raw_data.copy()
            self.TD = TD
            self.WL = WL
            self.base_dir = base_dir
            
            self._update_ui_limits_from_data()
            self.btn_run.setEnabled(True)
            self.label_status.setText(f"Loaded File: {len(self.WL)} WL, {len(self.TD)} TD")
            
        except Exception as e:
            QMessageBox.critical(self, "Error loading", str(e))

    def _clear_colorbar_if_exists(self, cbar):
        try:
            if cbar is not None:
                cbar.remove()
        except Exception:
            pass
    
# --- LÓGICA DE PROCESADO Y VISUALIZACIÓN ---

    def _preview_data_processing(self):
        """
        Toma data_raw, aplica Baseline -> Crop WL -> Crop Time -> Binning 
        y guarda el resultado en self.data_c para usarlo en el ajuste.
        """
        if self.data_raw is None: return
        
        temp_data = self.data_raw.copy()
        temp_WL = self.WL.copy()
        temp_TD = self.TD.copy()

        # 2. Baseline Correction
        n_pts = self.spin_bl.value()
        if n_pts > 0 and temp_data.shape[1] >= n_pts:
            # Asumiendo forma (WL, TD) -> axis 1 es tiempo
            baseline = np.mean(temp_data[:, :n_pts], axis=1, keepdims=True)
            temp_data = temp_data - baseline

        # 3. Crop Wavelength
        w_min = self.spin_wl_min.value()
        w_max = self.spin_wl_max.value()
        mask_w = (temp_WL >= min(w_min, w_max)) & (temp_WL <= max(w_min, w_max))
        
        if np.any(mask_w):
            temp_data = temp_data[mask_w, :]
            temp_WL = temp_WL[mask_w]

        # 4. Crop Time
        t_min = self.spin_t_min.value()
        t_max = self.spin_t_max.value()
        mask_t = (temp_TD >= min(t_min, t_max)) & (temp_TD <= max(t_min, t_max))
        
        if np.any(mask_t):
            temp_data = temp_data[:, mask_t]
            temp_TD = temp_TD[mask_t]

        # 5. Binning (Simple averaging)
        b_size = self.spin_bin.value()
        if b_size > 1:
            # Binning en eje espectral (WL)
            n_wl = temp_data.shape[0]
            new_len = n_wl // b_size
            if new_len > 0:
                # Recortamos el sobrante y hacemos reshape+mean
                temp_data = temp_data[:new_len*b_size, :]
                temp_data = temp_data.reshape(new_len, b_size, temp_data.shape[1]).mean(axis=1)
                temp_WL = temp_WL[:new_len*b_size]
                temp_WL = temp_WL.reshape(new_len, b_size).mean(axis=1)

        # GUARDAR RESULTADO PROCESADO
        self.data_c = temp_data
        
        # Guardamos versiones temporales de WL/TD para pintar correctamente
        self._wl_proc = temp_WL
        self._td_proc = temp_TD
        
        # Pintar
        self._update_exp_canvas(use_processed=True)
        self.label_status.setText(f"Data Ready: {len(temp_WL)} WL, {len(temp_TD)} TD")

    def _update_exp_canvas(self, use_processed=False):
            """Pinta el mapa experimental con escala dinámica y soporte Linear/SymLog."""
            if self.data_c is None: return
            
            self.ax_exp.clear()
            self._clear_colorbar_if_exists(self.cbar_exp)
            
            # Elegir qué ejes usar
            if use_processed and hasattr(self, '_wl_proc'):
                Xs = self._wl_proc
                Ys = self._td_proc
                Title = "Experimental (Processed)"
            else:
                Xs = self.WL
                Ys = self.TD
                Title = "Experimental (Raw)"
                
            # Protección ejes
            if Xs.shape[0] != self.data_c.shape[0] or Ys.shape[0] != self.data_c.shape[1]:
                Xs = np.arange(self.data_c.shape[0])
                Ys = np.arange(self.data_c.shape[1])
    
            try:
                vals = self.data_c.flatten()
                vmin = np.percentile(vals, 1) 
                vmax = np.percentile(vals, 99)
                
                self.pcm_exp = self.ax_exp.pcolormesh(Xs, Ys, self.data_c.T, 
                                                      shading="auto", cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_exp.set_title(Title)
                self.ax_exp.set_xlabel("Energy (eV)")
                self.ax_exp.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_exp.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_exp.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_exp)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_exp = self.canvas_exp.figure.colorbar(self.pcm_exp, cax=cax, label='Transient absorption / -')
                
                self.canvas_exp.draw_idle()
                
            except Exception as e:
                print(f"Plotting error: {e}")
    def _update_fit_canvas(self):
            """Pinta la reconstrucción con escala dinámica y soporte para Log/Linear."""
            if self.fit_fitres is None: return
    
            self.ax_fit.clear()
            self._clear_colorbar_if_exists(self.cbar_fit)
            
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_fitres.T 
    
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)  
                vmax = np.percentile(vals, 99) 
    
                self.pcm_fit = self.ax_fit.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet', 
                                                      vmin=vmin, vmax=vmax)
                
                self.ax_fit.set_title("Fit Reconstructed")
                self.ax_fit.set_xlabel("Energy (eV)")
                self.ax_fit.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_fit.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_fit.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_fit)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_fit = self.canvas_fit.figure.colorbar(self.pcm_fit, cax=cax, label='Transient absorption / -')
                self.canvas_fit.draw()
            except Exception as e:
                print(f"Error painting Fit: {e}")
        
    def _update_resid_canvas(self):
            """Pinta residuos con escala dinámica, JET y soporte para Log/Linear."""
            if self.fit_resid is None: return
    
            self.ax_resid.clear()
            self._clear_colorbar_if_exists(self.cbar_resid)
            
            Xs = getattr(self, '_wl_proc', self.WL)
            Ys = getattr(self, '_td_proc', self.TD)
            Z = self.fit_resid.T
    
            if Xs is None or Xs.shape[0] != Z.shape[1]: Xs = np.arange(Z.shape[1])
            if Ys is None or Ys.shape[0] != Z.shape[0]: Ys = np.arange(Z.shape[0])
    
            try:
                if Z.shape[0] < 2 or Z.shape[1] < 2: return
    
                vals = Z.flatten()
                vmin = np.percentile(vals, 1)
                vmax = np.percentile(vals, 99)
    
                self.pcm_resid = self.ax_resid.pcolormesh(Xs, Ys, Z, shading='auto', cmap='jet',
                                                          vmin=vmin, vmax=vmax)
                
                self.ax_resid.set_title("Residuals")
                self.ax_resid.set_xlabel("Energy (eV)")
                self.ax_resid.set_ylabel("Delay (ps)")
                
                # --- APLICAR ESCALA Y (CONDICIONAL) ---
                if hasattr(self, 'yscale') and self.yscale == 'symlog':
                    self.ax_resid.set_yscale('symlog', linthresh=1.0)
                else:
                    self.ax_resid.set_yscale('linear')
                
                divider = make_axes_locatable(self.ax_resid)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                self.cbar_resid = self.canvas_resid.figure.colorbar(self.pcm_resid, cax=cax, label='Residual')
                self.canvas_resid.draw()
            except Exception as e:
                print(f"Error painting Resid: {e}")
# --- LOGICA DEL AJUSTE (PIPELINE) ---
    def run_fit_pipeline(self):
            try:
                if self.data_raw is None:
                    QMessageBox.warning(self, "No data", "Load data first.")
                    return
    
                # 1. Preview y Procesado
                self._preview_data_processing()
                if self.data_c is None or self.data_c.size == 0: return
    
                # 2. Configuración
                self.numExp = self.spin_numExp.value()
                self.tech = self.combo_tech.currentText()
                self.t0_choice = 'Yes' if self.chk_chirp.isChecked() else 'No'
                
                # --- CORRECCIÓN AQUÍ: Detectar correctamente el tipo de modelo ---
                model_str = self.combo_model.currentText()
                if "Sequential" in model_str:
                    self.model_type = "Sequential"
                elif "Oscillation" in model_str:
                    self.model_type = "Dumped Oscillation"
                else:
                    self.model_type = "Parallel"
    
                # 3. GESTIÓN DE GUESSES (Validación de tamaño)
                if self.data_c is not None:
                    numWL = self.data_c.shape[0]
                else:
                    numWL = 0
                
                # Calcular L_needed según el modelo ACTUAL
                if self.model_type == "Dumped Oscillation":
                     # Estructura: [w, t0, taus..., alpha, omega, phi, (A..., B)_wl...]
                     # Params globales: 2 (w,t0) + numExp + 3 (osc) = 5 + numExp
                     # Params locales por WL: numExp + 1
                     if self.t0_choice == 'Yes':
                         # Chirp no implementado para oscilación, asumimos global
                         L_needed = (2 + self.numExp + 3) + numWL * (self.numExp + 1)
                     else:
                         L_needed = (2 + self.numExp + 3) + numWL * (self.numExp + 1)
                
                elif self.t0_choice == 'Yes': 
                    L_needed = 1 + self.numExp + numWL*(self.numExp+1)
                else:                       
                    L_needed = 2 + self.numExp + numWL*self.numExp
                
                # Si no hay guesses o el tamaño no coincide (porque cambiaste modelo/WL), regenerar
                if self.ini is None or len(self.ini) != L_needed:
                    print(f"Size mismatch (Vector: {len(self.ini) if self.ini is not None else 0}, Needed: {L_needed}). Regenerating defaults...")
                    self._generate_defaults()
                else:
                    print("Using existing guesses.")
    
                # 4. Ejecutar Ajuste
                self._temp_fit_TD = getattr(self, '_td_proc', self.TD)
                self._temp_fit_WL = getattr(self, '_wl_proc', self.WL)
                
                self._run_least_squares_with_progress()
                self._postprocess_fit_and_save()
    
            except Exception as e:
                QMessageBox.critical(self, "Fit Error", str(e))
                import traceback
                traceback.print_exc()

    def _open_guess_editor_and_update(self):
            """Abre la tabla de edición con ETIQUETAS DESCRIPTIVAS dinámicas."""
            numExp = self.spin_numExp.value()
            is_chirp = self.chk_chirp.isChecked()
            model_str = self.combo_model.currentText()
            is_oscillation = "Oscillation" in model_str
            
            # 1. Calcular longitud esperada y regenerar si es necesario
            if self.data_c is not None: numWL = self.data_c.shape[0]
            elif self.WL is not None: numWL = len(self.WL)
            else: numWL = 1
                
            if is_oscillation:
                L_needed = 2 + numExp + 3 + numWL * (numExp + 1)
            elif is_chirp:
                L_needed = 1 + numExp + numWL * (numExp + 1)
            else:
                L_needed = 2 + numExp + numWL * numExp
                
            if self.ini is None or len(self.ini) != L_needed:
                self._generate_defaults()
    
            # 2. Configurar Diálogo y Tabla
            L = len(self.ini)
            dlg = QDialog(self)
            dlg.setWindowTitle(f"Edit Initial Guesses - {model_str}")
            dlg.resize(800, 600)
            v = QVBoxLayout()
            
            table = QTableWidget(L, 5)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.setHorizontalHeaderLabels(["Parameter", "Value", "Lower Bound", "Upper Bound", "Fix?"])
            
            if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L:
                self.is_fixed = np.zeros(L, dtype=bool)
                
            # --- BUCLE DE LLENADO CON ETIQUETAS INTELIGENTES ---
            for i in range(L):
                label = f"{i}: "
                
                if is_oscillation:
                    # Lógica de etiquetas para el modelo de OSCILACIÓN
                    if i == 0: label += "w (IRF Width)"
                    elif i == 1: label += "t0 (Time Zero)"
                    elif i < 2 + numExp: 
                        label += f"τ{i-1} (Lifetime)"
                    elif i == 2 + numExp: 
                        label += "α (Damping/Decay)"
                    elif i == 2 + numExp + 1: 
                        label += "ω (Ang. Frequency)"
                    elif i == 2 + numExp + 2: 
                        label += "φ (Phase)"
                    else:
                        # Parámetros locales [A1, A2... An, B]
                        local_idx = i - (2 + numExp + 3)
                        wl_idx = local_idx // (numExp + 1)
                        p_idx = local_idx % (numExp + 1)
                        curr_wl = self._wl_proc[wl_idx] if hasattr(self, '_wl_proc') else wl_idx
                        if p_idx < numExp:
                            label += f"A{p_idx+1} (Amp) @ {curr_wl:.1f}nm"
                        else:
                            label += f"B (Osc. Amp) @ {curr_wl:.1f}nm"
                
                else:
                    # Lógica ORIGINAL para Parallel/Sequential/Chirp
                    if not is_chirp:
                        if i == 0: label += "w (IRF Width)"
                        elif i == 1: label += "t0 (Time Zero)"
                        elif i < 2 + numExp: label += f"τ{i-1} (Lifetime)"
                        else:
                            local_idx = i - (2 + numExp)
                            wl_idx = local_idx // numExp
                            p_idx = local_idx % numExp
                            label += f"A{p_idx+1} @ WL {wl_idx}"
                    else:
                        if i == 0: label += "w (IRF Width)"
                        elif i < 1 + numExp: label += f"τ{i} (Lifetime)"
                        else:
                            label += "Local (t0 or Amp)"
    
                # Llenar la fila de la tabla
                item_lbl = QTableWidgetItem(label)
                item_lbl.setFlags(item_lbl.flags() ^ Qt.ItemIsEditable)
                table.setItem(i, 0, item_lbl)
                table.setItem(i, 1, QTableWidgetItem(str(self.ini[i])))
                table.setItem(i, 2, QTableWidgetItem(str(self.limi[i])))
                table.setItem(i, 3, QTableWidgetItem(str(self.lims[i])))
                
                chk_item = QTableWidgetItem()
                chk_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                chk_item.setCheckState(Qt.Checked if self.is_fixed[i] else Qt.Unchecked)
                table.setItem(i, 4, chk_item)
    
            v.addWidget(table)
            
            # Botones de control
            btns = QHBoxLayout()
            btn_reset = QPushButton("Reset to Defaults")
            btn_reset.clicked.connect(lambda: [self._generate_defaults(), dlg.accept(), self._open_guess_editor_and_update()])
            btn_ok = QPushButton("Save & Close")
            btn_ok.clicked.connect(dlg.accept)
            btns.addWidget(btn_reset); btns.addWidget(btn_ok)
            v.addLayout(btns)
            
            dlg.setLayout(v)
            if dlg.exec_() == QDialog.Accepted:
                for i in range(L):
                    self.ini[i] = float(table.item(i, 1).text())
                    self.limi[i] = float(table.item(i, 2).text())
                    self.lims[i] = float(table.item(i, 3).text())
                    self.is_fixed[i] = (table.item(i, 4).checkState() == Qt.Checked)
    def _run_least_squares_with_progress(self):
        
        TD = self._temp_fit_TD
        WL = self._temp_fit_WL
        numWL = len(WL)
        data_flat = self.data_c.T.flatten()
    
        if not hasattr(self, 'is_fixed') or len(self.is_fixed) != len(self.ini):
            self.is_fixed = np.zeros(len(self.ini), dtype=bool)
            
        free_indices = np.where(~self.is_fixed)[0]
        x0_free = self.ini[free_indices]
        low_free = self.limi[free_indices]
        upp_free = self.lims[free_indices]
    
        # --- LÓGICA DE PROGRESO ---
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Iterating: %v") # Muestra el número de iteración
        self.iter_count = 0 
    
        def residuals(p_free):
            # 1. Incrementar contador cada vez que se evalúa el modelo
            self.iter_count += 1
            
            # 2. Actualizar la barra cada N evaluaciones para no ralentizar demasiado
            if self.iter_count % 10 == 0:
                # Como no sabemos el total, hacemos que la barra "baile" (modo indefinido)
                # o simplemente mostramos el conteo de iteraciones
                val = (self.iter_count // 10) % 101
                self.progress_bar.setValue(val)
                
                # ¡ESTO ES LO MÁS IMPORTANTE! 
                # Fuerza a la interfaz a procesar los cambios de diseño y botones
                QApplication.processEvents()
    
            x_full = self.ini.copy()
            x_full[free_indices] = p_free
            
            if self.model_type == "Sequential":
                F = fit.eval_sequential_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            elif self.model_type == 'Dumped Oscillation':
                F = fit.eval_oscillation_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            else:
                F = fit.eval_global_model(x_full, TD, self.numExp, numWL, self.t0_choice)
            
            return F.flatten() - data_flat
    
        try:
            res = least_squares(
                fun=residuals,
                x0=x0_free,
                bounds=(low_free, upp_free),
                method='trf',
                verbose=0
            )
            
            self.fit_result = res
            self.fit_x = self.ini.copy()
            self.fit_x[free_indices] = res.x
            
            # Al terminar, ponemos la barra al 100%
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("Fit Completed")
            
        except Exception as e:
            self.progress_bar.setValue(0)
            raise e

    def _postprocess_fit_and_save(self):
            """Calcula estadísticas, extrae espectros con errores y guarda archivos en /fit/."""
            import fit
            import os
            import numpy as np
            from PyQt5.QtWidgets import QMessageBox
    
            if self.fit_result is None:
                return
    
            x = self.fit_x
            TD = getattr(self, '_temp_fit_TD', self.TD)
            WL = getattr(self, '_temp_fit_WL', self.WL)
            
            if TD is None or WL is None:
                print("Error: No se encontraron los ejes (TD/WL) del ajuste.")
                return
    
            numWL = len(WL)
            numExp = self.numExp
    
            # --- 1. Reconstruir Matriz de Ajuste y Residuos ---
            if self.model_type == "Sequential":
                F_mat = fit.eval_sequential_model(x, TD, numExp, numWL, self.t0_choice)
            elif self.model_type == 'Dumped Oscillation':
                F_mat = fit.eval_oscillation_model(x, TD, numExp, numWL, self.t0_choice)
            else:
                F_mat = fit.eval_global_model(x, TD, numExp, numWL, self.t0_choice)
                
            fitres = F_mat.T 
            resid = self.data_c - fitres
            
            self.fit_fitres = fitres
            self.fit_resid = resid
    
            # --- 2. Cálculo de Errores (CI) ROBUSTO ---
            L_total = len(x)
            self.ci = np.zeros(L_total) # Por defecto 0
            
            try:
                # Asegurar que is_fixed existe
                if not hasattr(self, 'is_fixed') or len(self.is_fixed) != L_total:
                    self.is_fixed = np.zeros(L_total, dtype=bool)
                
                free_indices = np.where(~self.is_fixed)[0]
                J = self.fit_result.jac 
                
                # Solo calcular si hay parámetros libres y Jacobiano válido
                if J is not None and J.size > 0 and len(free_indices) > 0:
                    # Usa Pseudo-Inversa (pinv) en lugar de inv para evitar crash por matriz singular
                    # cov_free = inv(J.T * J) * MSE
                    H = J.T @ J
                    cov_free = np.linalg.pinv(H) 
                    
                    # Grados de libertad
                    dof = resid.size - len(free_indices)
                    if dof > 0:
                        mse = np.sum(resid**2) / dof
                        # Diagonal de la covarianza * MSE = Varianza de los parámetros
                        var_free = np.diagonal(cov_free) * mse
                        # Evitar raíces de negativos por errores numéricos pequeños
                        err_free = np.sqrt(np.maximum(var_free, 0))
                        
                        # Mapear errores calculados a sus posiciones
                        self.ci[free_indices] = err_free
                    else:
                        print("Warning: Degrees of freedom <= 0. Cannot compute errors.")
    
            except Exception as e:
                # Mostrar el error real en consola para depuración
                print(f"CRITICAL ERROR calculating covariance: {e}")
                # Opcional: Avisar al usuario con un popup si falla gravemente
                # QMessageBox.warning(self, "Stats Error", f"Could not compute errors:\n{e}")
    
            # --- 3. Extraer Taus y sus Errores ---
            # (El resto del código se mantiene igual, pero lo incluyo para que copies/pegues seguro)
            idx_tau = 1 if self.t0_choice == 'Yes' else 2
            
            # Protección de índices por si el modelo cambia
            end_tau = idx_tau + numExp
            if end_tau <= len(x):
                self.extracted_taus = x[idx_tau : end_tau]
                self.extracted_errtaus = self.ci[idx_tau : end_tau]
            else:
                self.extracted_taus = np.zeros(numExp)
                self.extracted_errtaus = np.zeros(numExp)
    
            # --- 4. Extraer Amplitudes y sus Errores ---
            self.As = np.zeros((numExp, numWL))
            self.errAs = np.zeros((numExp, numWL))
            self.Bs = None      # To store Oscillation Amplitude Spectrum
            self.errBs = None
            
            try:
                if self.t0_choice == 'No':
                    if "Oscillation" in self.model_type:
                        # Structure: [w, t0, taus..., alpha, omega, phi, ...]
                        base_A = 2 + numExp + 3
                        params_per_wl = numExp + 1 
                        
                        # Extract all local params (A's + B)
                        all_local = x[base_A:]
                        all_local_err = self.ci[base_A:]
                        
                        # Reshape to (numWL, params_per_wl)
                        mat_local = all_local.reshape(numWL, params_per_wl)
                        mat_err = all_local_err.reshape(numWL, params_per_wl)
                        
                        # Separate A's (Decays) and B (Oscillation)
                        self.As = mat_local[:, :numExp].T
                        self.errAs = mat_err[:, :numExp].T
                        
                        # The last column is B (Oscillation Amplitude)
                        self.Bs = mat_local[:, numExp]
                        self.errBs = mat_err[:, numExp]
                        
                        self.t0s = np.full(numWL, x[1])
    
                    else:
                        # Standard Model logic...
                        base_A = 2 + numExp
                        self.As = x[base_A:].reshape(numWL, numExp).T
                        self.errAs = self.ci[base_A:].reshape(numWL, numExp).T
                        self.t0s = np.full(numWL, x[1])
                else:
                    pass
                    
            except Exception as e:
                print(f"Error extrayendo amplitudes: {e}")
    
            # --- 5. Guardado (Igual que antes) ---
            base_dir = self.base_dir
            outdir = os.path.join(base_dir, "fit")
            os.makedirs(outdir, exist_ok=True)
    
            try:
                np.save(os.path.join(outdir, "GFitResults.npy"), {
                    "taus": self.extracted_taus,
                    "err_taus": self.extracted_errtaus,
                    "As": self.As,
                    "errAs": self.errAs,
                    "WL": WL,
                    "TD": TD,
                    "fitres": fitres,
                    "resid": resid
                })
    
                np.savetxt(os.path.join(outdir, "WL.txt"), WL, fmt='%.6f', header="Wavelength (nm)")
                np.savetxt(os.path.join(outdir, "TD.txt"), TD, fmt='%.6f', header="Time Delay (ps)")
                
                with open(os.path.join(outdir, "Amplitudes.txt"), 'w') as f:
                    header_list = [f"A{i+1}\tErrA{i+1}" for i in range(numExp)]
                    f.write("WL(nm)\t" + "\t".join(header_list) + "\n")
                    for i in range(numWL):
                        line_data = [f"{WL[i]:.2f}"]
                        for j in range(numExp):
                            val = self.As[j, i] if j < self.As.shape[0] else 0
                            err = self.errAs[j, i] if j < self.errAs.shape[0] else 0
                            line_data.append(f"{val:.6e}")
                            line_data.append(f"{err:.6e}")
                        f.write("\t".join(line_data) + "\n")
         
                print(f"Resultados exportados a: {outdir}")
    
            except Exception as e:
                print(f"Error guardando archivos: {e}")     
    
            self._update_fit_canvas()
            self._update_resid_canvas()
            self.btn_show_das.setEnabled(True)
    
            rmsd = np.sqrt(np.mean(resid**2))
            QMessageBox.information(self, "Ajuste Finalizado", 
                                    f"Optimización completada.\nRMSD: {rmsd:.2e}\nDatos guardados en /fit/")
    def plot_das_and_more(self):
            """
            Abre ventana externa con DAS/SAS.
            Si hay oscilación, separa los gráficos en dos paneles.
            """
            if self.As is None: return
    
            # Definir directorio de salida para plots
            outdir = os.path.join(self.base_dir, "Plots")
            os.makedirs(outdir, exist_ok=True)
            
            wl = getattr(self, '_wl_proc', self.WL)
            td = getattr(self, '_td_proc', self.TD)
            
            # --- DETECTAR SI HAY OSCILACIÓN ---
            has_oscillation = hasattr(self, 'Bs') and self.Bs is not None
            
            # Configurar figura: 2 paneles si hay oscilación, 1 si no
            if has_oscillation:
                fig_das, (ax_das, ax_osc) = plt.subplots(1, 2, figsize=(14, 6))
            else:
                fig_das, ax_das = plt.subplots(figsize=(8, 6))
                ax_osc = None
    
            # --- 1. PLOT DAS (Exponenciales) ---
            colors = ['b', 'r', 'g', 'orange', 'm', 'c']
            
            for n in range(self.numExp):
                tau_val = self.extracted_taus[n]
                # Verificar si existe error y no es NaN
                if self.extracted_errtaus is not None and n < len(self.extracted_errtaus):
                     err_tau = self.extracted_errtaus[n]
                     if np.isnan(err_tau): err_tau = 0.0
                else:
                     err_tau = 0.0
    
                lbl = f"τ{n+1} = {tau_val:.2f} ± {err_tau:.2f} ps"
                color = colors[n % len(colors)]
    
                # Línea principal
                ax_das.plot(wl, self.As[n], label=lbl, color=color, linewidth=2)
                
                # Sombra de error (Si existe)
                if self.errAs is not None:
                    lower = np.nan_to_num(self.As[n] - self.errAs[n])
                    upper = np.nan_to_num(self.As[n] + self.errAs[n])
                    ax_das.fill_between(wl, lower, upper, color=color, alpha=0.2)
            
            ax_das.set_xlabel("Wavelength (nm)")
            if self.model_type == "Sequential":
                ax_das.set_ylabel("SAS (Concentration)")
                ax_das.set_title("Species Associated Spectra (SAS)")
            else:
                ax_das.set_ylabel("DAS Amplitude (ΔA)")
                ax_das.set_title("Decay Associated Spectra (DAS)")
            
            ax_das.legend()
            ax_das.axhline(0, color='k', linestyle='--', alpha=0.5)
            ax_das.grid(True, linestyle=':', alpha=0.4)
    
            # --- 2. PLOT OSCILACIÓN (Si existe) ---
            if has_oscillation and ax_osc is not None:
                # Recuperamos parámetros físicos del vector fit_x
                # Índices: [w, t0, tau1..n, alpha, omega, phi, ...]
                alpha = self.fit_x[2 + self.numExp]
                omega = self.fit_x[2 + self.numExp + 1]
                phi   = self.fit_x[2 + self.numExp + 2]
                
                # Crear título informativo
                title_osc = (f"Oscillation Spectrum\n"
                             f"Damping α={alpha:.4f} | Freq ω={omega:.4f} | Phase φ={phi:.2f}")
                
                # Plot Spectrum B
                ax_osc.plot(wl, self.Bs, color='black', linewidth=2, label='Oscillation Amplitude (B)')
                
                # Error del espectro B
                if self.errBs is not None:
                    ax_osc.fill_between(wl, self.Bs - self.errBs, self.Bs + self.errBs, color='black', alpha=0.1)
                
                ax_osc.set_xlabel("Wavelength (nm)")
                ax_osc.set_ylabel("Oscillation Amplitude")
                ax_osc.set_title(title_osc, color='darkblue') # Color para resaltar
                ax_osc.axhline(0, color='k', linestyle='--', alpha=0.5)
                ax_osc.grid(True, linestyle=':', alpha=0.4)
                ax_osc.legend()
    
            fig_das.tight_layout()
    
            # Guardar imagen
            savename = "DAS_and_Oscillation.png" if has_oscillation else "DAS.png"
            try:
                fig_das.savefig(os.path.join(outdir, savename), dpi=300)
                print(f"Plot saved to {outdir}")
            except Exception as e:
                print(f"Error saving DAS plot: {e}")
    
            fig_das.show()
            
            fig_res, ax_res = plt.subplots()
            pcm = ax_res.pcolormesh(wl, td, self.fit_resid.T, cmap='jet', shading='auto')
            fig_res.colorbar(pcm, ax=ax_res, label='Residuals')
            ax_res.set_title("Residuals Map")
            ax_res.set_xlabel("Energy (eV)")
            ax_res.set_ylabel("Delay (ps)")
            if hasattr(self, 'yscale') and self.yscale == 'symlog':
                 ax_res.set_yscale('symlog', linthresh=1.0)
            fig_res.tight_layout()
            fig_res.savefig(os.path.join(outdir, "Residuals_Map.png"), dpi=300)
            plt.close(fig_res)
    
            cont = True
            while cont:
                text_default = f"{wl[len(wl)//2]:.1f}"
                wl_str, ok = QInputDialog.getText(self, "Check Trace", 
                                                  f"Enter wavelength nm ({wl.min():.1f}-{wl.max():.1f}):", 
                                                  text=text_default)
                if not ok: break
    
                try:
                    target_wl = float(wl_str)
                    idx = np.argmin(np.abs(wl - target_wl))
                    real_wl = wl[idx]
    
                    y_exp = self.data_c[idx, :]
                    y_fit = self.fit_fitres[idx, :]
    
                    # Crear figura Trace
                    fig_trace, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
                    fig_trace.suptitle(f"Fit at {real_wl:.1f} nm", fontsize=14)
    
                    # Linear
                    ax1.plot(td, y_exp, 'bo', markersize=4, alpha=0.6, label='Data')
                    ax1.plot(td, y_fit, 'r-', linewidth=2, label='Fit')
                    ax1.set_xlabel("Time / ps")
                    ax1.set_ylabel("ΔA")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
    
                    # Log
                    mask_pos = td > 0
                    if np.any(mask_pos):
                        ax2.plot(td[mask_pos], y_exp[mask_pos], 'bo', markersize=4, alpha=0.6)
                        ax2.plot(td[mask_pos], y_fit[mask_pos], 'r-', linewidth=2)
                        ax2.set_xscale('log')
                        ax2.set_xlabel("Time / ps (log scale)")
                        ax2.grid(True, which="both", ls="-", alpha=0.3)
    
                    plt.tight_layout()
                    plt.show(block=True) 
    
                    # Guardar Traza Individual?
                    resp = QMessageBox.question(self, "Save Trace?",
                                                f"¿Deseas guardar los archivos de la traza a {real_wl:.1f} nm?",
                                                QMessageBox.Yes | QMessageBox.No)
    
                    if resp == QMessageBox.Yes:
                        img_name = f"Trace_{real_wl:.1f}nm.png"
                        fig_trace.savefig(os.path.join(outdir, img_name), dpi=300)
    
                        txt_name = f"Fit_{real_wl:.1f}nm.txt"
                        txt_path = os.path.join(outdir, txt_name)
                        
                        data_stack = np.column_stack((td, y_exp, y_fit))
                        header_txt = "TD(ps)\tExp(A)\tFit(A)"
                        
                        np.savetxt(txt_path, data_stack, fmt='%1.6e', delimiter='\t',
                                   header=header_txt, comments='# ')
    
                    plt.close(fig_trace)
    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al procesar la traza: {e}")
    
                if QMessageBox.question(self, "Continuar", "¿Ver otra traza?", 
                                        QMessageBox.Yes|QMessageBox.No) == QMessageBox.No:
                    cont = False