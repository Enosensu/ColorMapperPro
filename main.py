import sys
import os
import ctypes

# ==========================================
# [关键修复] 强制添加 DLL 搜索路径
# ==========================================
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
    internal_path = os.path.join(application_path, '_internal')
    os.environ['PATH'] = internal_path + os.pathsep + os.environ.get('PATH', '')
    if os.path.exists(internal_path) and hasattr(os, 'add_dll_directory'):
        try: os.add_dll_directory(internal_path)
        except Exception as e: print(f"Warning: Could not add DLL directory: {e}")

# ==========================================
# 尝试导入 CuPy
# ==========================================
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    HAS_CUPY = True
    print(f">>> GPU 加速已成功启用 (CuPy)")
except Exception as e:
    HAS_CUPY = False
    print(f">>> 错误: 无法加载 CuPy，将回退到 CPU: {e}")

import cv2
import numpy as np
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QFrame, 
                             QGroupBox, QMessageBox, QSpinBox, QSplitter,
                             QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, 
                             QAbstractItemView, QProgressBar, QComboBox, QSizePolicy)
# [修复] 引入 QTimer
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QPoint, QSize, QTimer
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent, 
                         QIcon, QPen, QBrush, QKeySequence, QShortcut, QPolygonF, QTransform, QCursor)

# ==========================================
# 0. 系统配置
# ==========================================
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")

try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('mycompany.color.mapper.pro.v15')
except: pass

# ==========================================
# 1. 核心算法
# ==========================================
class ImageProcessor:
    def __init__(self):
        self.use_gpu = False
        if HAS_CUPY:
            try:
                cp.array([1]).device
                self.use_gpu = True
                print(f">>> GPU 加速已启用: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')}")
            except: self.use_gpu = False

    def recommend_k(self, img, max_search_k=24):
        if img is None: return 16
        h, w = img.shape[:2]
        scale = min(100.0/h, 100.0/w) 
        small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
        data = lab.reshape(-1, 3).astype(np.float32)
        unique_colors = len(np.unique(data.astype(int), axis=0))
        upper = min(max_search_k, unique_colors)
        if upper < 3: return max(2, unique_colors)
        
        distortions = []
        K_range = list(range(2, upper + 1, 2))
        if len(K_range) < 2: K_range = [2, upper]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        for k in K_range:
            if k >= unique_colors: distortions.append(0); continue
            d, _, _ = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            distortions.append(d)
            
        p1 = np.array([K_range[0], distortions[0], 0])
        p2 = np.array([K_range[-1], distortions[-1], 0])
        max_dist = 0; best_k = K_range[0]
        vec_line = p2 - p1; norm_line = np.linalg.norm(vec_line)
        if norm_line == 0: return 16
        
        for i, k in enumerate(K_range):
            p = np.array([k, distortions[i], 0])
            d = np.linalg.norm(np.cross(vec_line, p - p1)) / norm_line
            if d > max_dist: max_dist = d; best_k = k
            
        final_k = int(best_k * 1.2)
        final_k = min(final_k, upper)
        return int(max(2, final_k + (1 if final_k % 2 != 0 else 0)))

    def get_dominant_colors(self, img, k):
        if img is None: return np.array([])
        h, w = img.shape[:2]
        factor = max(1, max(h, w) // 150)
        small = img[::factor, ::factor]
        data = small.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        except: return np.array([])
        labels = labels.flatten()
        counts = np.bincount(labels)
        return centers[np.argsort(counts)[::-1]].astype(np.uint8)

    def recommend_unique_n(self, img):
        if img is None: return 16
        h, w = img.shape[:2]
        scale = min(200.0/h, 200.0/w)
        small = cv2.resize(img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale < 1.0 else img
        pixels = small.reshape(-1, 3)
        _, counts = np.unique(pixels, axis=0, return_counts=True)
        counts.sort()
        cumulative = np.cumsum(counts[::-1])
        n = np.searchsorted(cumulative, small.shape[0]*small.shape[1] * 0.85) + 1
        return int(max(2, min(n, 64)))

    def get_top_unique_colors(self, img, n):
        if img is None: return np.array([])
        h, w = img.shape[:2]
        if h*w > 1000000:
            s = np.sqrt(1000000/(h*w))
            img = cv2.resize(img, (0,0), fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        pixels = img.reshape(-1, 3)
        u_colors, counts = np.unique(pixels, axis=0, return_counts=True)
        if len(u_colors) <= n: return u_colors[np.argsort(counts)[::-1]].astype(np.uint8)
        top_idx = np.argpartition(counts, -n)[-n:]
        return u_colors[top_idx[np.argsort(counts[top_idx])[::-1]]].astype(np.uint8)

    def color_transfer(self, target_img, palette, color_space='LAB'):
        if target_img is None or len(palette) == 0: return target_img
        
        cvt_in = None
        if color_space == 'LAB': cvt_in = cv2.COLOR_BGR2Lab
        elif color_space == 'HSV': cvt_in = cv2.COLOR_BGR2HSV
            
        if cvt_in is not None:
            src_data = cv2.cvtColor(target_img, cvt_in)
            pal_bgr = palette.reshape(1, -1, 3).astype(np.uint8)
            pal_conv = cv2.cvtColor(pal_bgr, cvt_in).reshape(-1, 3).astype(np.float32)
        else:
            src_data = target_img
            pal_conv = palette.astype(np.float32)

        pal_final_bgr = palette.astype(np.uint8)
        h, w = src_data.shape[:2]

        if self.use_gpu:
            try:
                img_gpu = cp.asarray(src_data.reshape(-1, 3).astype(np.float32))
                pal_gpu = cp.asarray(pal_conv)
                pal_bgr_gpu = cp.asarray(pal_final_bgr)

                N = img_gpu.shape[0]
                chunk_size = 1000000 
                pal_gpu_expanded = pal_gpu[cp.newaxis, :, :]
                nearest_indices = cp.empty(N, dtype=cp.int32)

                for i in range(0, N, chunk_size):
                    end = min(i + chunk_size, N)
                    img_chunk = img_gpu[i:end][:, cp.newaxis, :]
                    diff = img_chunk - pal_gpu_expanded
                    dists = cp.sum(diff**2, axis=2)
                    nearest_indices[i:end] = cp.argmin(dists, axis=1)
                
                res_flat_gpu = pal_bgr_gpu[nearest_indices]
                res_img = cp.asnumpy(res_flat_gpu).reshape(h, w, 3).astype(np.uint8)
                return res_img
            except Exception as e:
                print(f"GPU Error: {e}, fallback to CPU")
        
        # CPU Fallback
        img_flat = src_data.reshape(-1, 3).astype(np.int32)
        pal_conv_int = pal_conv.astype(np.int32)
        u_colors, inverse_indices = np.unique(img_flat, axis=0, return_inverse=True)
        mapped_u = np.zeros_like(u_colors, dtype=np.uint8)
        chunk_size = 2000 
        pal_tensor = pal_conv_int[np.newaxis, :, :]
        
        for i in range(0, len(u_colors), chunk_size):
            chunk = u_colors[i:i+chunk_size]
            diff = chunk[:, np.newaxis, :] - pal_tensor
            dists = np.sum(diff**2, axis=2)
            idx = np.argmin(dists, axis=1)
            mapped_u[i:i+chunk_size] = pal_final_bgr[idx]
            
        res_flat = mapped_u[inverse_indices]
        return res_flat.reshape(h, w, 3).astype(np.uint8)

PROCESSOR = ImageProcessor()

# ==========================================
# 2. 资源管理
# ==========================================
class ResourceManager:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="gemini_res_")
        self.icons = {}
        self._create_svg_files()

    def _create_svg_files(self):
        svg_up = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12"><path fill="#ffffff" d="M6 3 L10 9 L2 9 Z"/></svg>'
        svg_down = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12"><path fill="#ffffff" d="M6 9 L2 3 L10 3 Z"/></svg>'
        p_up = os.path.join(self.temp_dir, "up.svg")
        p_down = os.path.join(self.temp_dir, "down.svg")
        with open(p_up, "w") as f: f.write(svg_up)
        with open(p_down, "w") as f: f.write(svg_down)
        self.icons['up'] = p_up.replace("\\", "/")
        self.icons['down'] = p_down.replace("\\", "/")

    def get_icon_url(self, name):
        return f'url("{self.icons.get(name, "")}")'

    def cleanup(self):
        try: shutil.rmtree(self.temp_dir)
        except: pass

RES_MANAGER = ResourceManager()
ICON_CHECK_WHITE = 'url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAbklEQVQ4je2S0Q3AIAxD78M6Q3dI958u0g3Sqf+0Ug0i+sA5X6wEcOzYJAn8M2RInz1CDsCR09puvZaF/CAlhHnN77y2CwBExN3+rNfS6iBvB3k7yN9B/g7+d/C/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g/8d/O/g7yDvIC7wB/k0315YVvOAAAAAASUVORK5CYII=")'
ICON_UP_PATH = RES_MANAGER.get_icon_url("up")
ICON_DOWN_PATH = RES_MANAGER.get_icon_url("down")

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# 3. UI 组件
# ==========================================
class DraggableColorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(162, 100)
        self.is_dragging = False; self.drag_start_pos = QPoint()
        self.row_orig = self.create_color_row("原图")
        self.row_res = self.create_color_row("结果")
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.container = QFrame(); self.container.setObjectName("ColorPanelFrame")
        self.container.setStyleSheet("#ColorPanelFrame { background-color: rgba(30, 30, 30, 220); border: 1px solid #555; border-radius: 8px; } QLabel { background-color: transparent; }")
        inner = QVBoxLayout(self.container); inner.setSpacing(5)
        title = QLabel("像素对比"); title.setStyleSheet("color:#aaa; font-size:10px; font-weight:bold;"); title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(title)
        inner.addLayout(self.row_orig['layout'])
        inner.addLayout(self.row_res['layout'])
        layout.addWidget(self.container)

    def create_color_row(self, text):
        l = QHBoxLayout()
        lbl = QLabel(text); lbl.setStyleSheet("color:#ccc; font-weight:bold; font-size:11px;"); lbl.setFixedWidth(40)
        swatch = QLabel(); swatch.setFixedSize(30, 16); swatch.setStyleSheet("border:1px solid #555;")
        val = QLabel("-------"); val.setStyleSheet("color:#fff; font-family:Consolas; font-size:11px;")
        l.addWidget(lbl); l.addWidget(swatch); l.addWidget(val); l.addStretch()
        return {'layout': l, 'swatch': swatch, 'val': val}

    def update_colors(self, c_orig, c_res):
        def set_c(row, bgr):
            if bgr is not None:
                hex_s = QColor(int(bgr[2]), int(bgr[1]), int(bgr[0])).name().upper()
                row['swatch'].setStyleSheet(f"background-color:{hex_s}; border:1px solid #888;")
                row['val'].setText(hex_s)
            else:
                row['swatch'].setStyleSheet("background-color:transparent; border:1px solid #555;")
                row['val'].setText("-------")
        set_c(self.row_orig, c_orig)
        set_c(self.row_res, c_res)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True; self.drag_start_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
    def mouseMoveEvent(self, e):
        if self.is_dragging: self.move(e.globalPosition().toPoint() - self.drag_start_pos)
    def mouseReleaseEvent(self, e): self.is_dragging = False

class OverlayControls(QWidget):
    toggle_signal = pyqtSignal(bool); reset_signal = pyqtSignal(); zoom_100_signal = pyqtSignal()
    def __init__(self, parent=None, show_toggle=True):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(5)
        btn_style = """
            QPushButton { background-color: rgba(40, 40, 40, 200); color: white; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; font-size: 12px; }
            QPushButton:hover { background-color: rgba(60, 60, 60, 230); border-color: #0078d4; }
            QPushButton:checked { background-color: #0078d4; border-color: #0078d4; }
            QPushButton:pressed { background-color: #0078d4; border-color: #0078d4; color: white; }
        """
        self.btn_reset = QPushButton("复位"); self.btn_reset.setStyleSheet(btn_style); self.btn_reset.clicked.connect(self.reset_signal.emit)
        self.btn_100 = QPushButton("100%"); self.btn_100.setStyleSheet(btn_style); self.btn_100.clicked.connect(self.zoom_100_signal.emit)
        layout.addWidget(self.btn_reset); layout.addWidget(self.btn_100)
        if show_toggle:
            self.btn_toggle = QPushButton("当前: 原图"); self.btn_toggle.setCheckable(True); self.btn_toggle.setStyleSheet(btn_style)
            self.btn_toggle.clicked.connect(self.on_toggle)
            layout.addWidget(self.btn_toggle)
        else:
            self.btn_toggle = None

    def on_toggle(self):
        checked = self.btn_toggle.isChecked()
        self.btn_toggle.setText("当前: 结果" if checked else "当前: 原图")
        self.toggle_signal.emit(checked)
    
    def set_toggle_state(self, checked):
        if self.btn_toggle:
            self.btn_toggle.setChecked(checked)
            self.btn_toggle.setText("当前: 结果" if checked else "当前: 原图")

class UndoControls(QWidget):
    undo = pyqtSignal(); redo = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(5)
        btn_style = """
            QPushButton { background-color: rgba(40, 40, 40, 200); color: white; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; font-size: 12px; }
            QPushButton:hover { background-color: rgba(60, 60, 60, 230); border-color: #0078d4; }
            QPushButton:pressed { background-color: #0078d4; border-color: #0078d4; color: white; }
            QPushButton:disabled { background-color: rgba(30, 30, 30, 150); color: #888; border-color: #444; }
        """
        self.b_undo = QPushButton("↩"); self.b_undo.setToolTip("撤销 (Ctrl+Z)"); self.b_undo.setStyleSheet(btn_style); self.b_undo.clicked.connect(self.undo.emit); self.b_undo.setEnabled(False)
        self.b_redo = QPushButton("↪"); self.b_redo.setToolTip("重做 (Ctrl+Shift+Z/Ctrl+Y)"); self.b_redo.setStyleSheet(btn_style); self.b_redo.clicked.connect(self.redo.emit); self.b_redo.setEnabled(False)
        layout.addWidget(self.b_undo); layout.addWidget(self.b_redo)
    def update_states(self, u, r):
        self.b_undo.setEnabled(u); self.b_redo.setEnabled(r)

# ==========================================
# 4. 图像预览器
# ==========================================
class BaseImageViewer(QWidget):
    pixel_hover = pyqtSignal(object, object) 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_pixmap = None; self.scale_factor = 1.0; self.offset = QPointF(0, 0)
        self.is_dragging = False; self.last_mouse_pos = QPointF(0, 0)
        self.setMouseTracking(True); self.highlight_mask = None; self.dim_mask = None 
    def set_image_data(self, cv_img):
        if cv_img is None: self.img_pixmap = None; self.update(); return
        h, w, c = cv_img.shape; bytes_pl = c * w
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qim = QImage(rgb.data, w, h, bytes_pl, QImage.Format.Format_RGB888)
        self.img_pixmap = QPixmap.fromImage(qim); self.update()
    def set_highlight(self, mask): self.highlight_mask = mask; self.update()
    def set_dim_mask(self, mask): self.dim_mask = mask; self.update()
    def reset_view(self): self.offset = QPointF(0,0); self.update()
    def zoom_100(self): self.scale_factor = 1.0; self.offset = QPointF(0,0); self.update()
    def get_img_coords(self, pos):
        if not self.img_pixmap: return -1, -1
        cx, cy = self.width()/2, self.height()/2
        lx = (pos.x() - (cx + self.offset.x())) / self.scale_factor
        ly = (pos.y() - (cy + self.offset.y())) / self.scale_factor
        img_x = int(lx + self.img_pixmap.width()/2); img_y = int(ly + self.img_pixmap.height()/2)
        return img_x, img_y
    def paintEvent(self, event):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing, False); p.fillRect(self.rect(), QColor("#1e1e1e"))
        if self.img_pixmap and not self.img_pixmap.isNull():
            cx, cy = self.width()/2, self.height()/2
            p.translate(cx + self.offset.x(), cy + self.offset.y()); p.scale(self.scale_factor, self.scale_factor)
            x = -self.img_pixmap.width()/2; y = -self.img_pixmap.height()/2
            p.drawPixmap(QPointF(x, y), self.img_pixmap)
            pen = QPen(QColor(255, 255, 255, 50)); pen.setWidthF(1.0/self.scale_factor); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x, y, self.img_pixmap.width(), self.img_pixmap.height()))
            if self.dim_mask is not None:
                p.save(); self._apply_dimming(p, self.dim_mask, x, y); p.restore()
            if self.highlight_mask is not None:
                self._draw_mask_outline(p, self.highlight_mask, QColor(0, 255, 255, 255), 2.0/self.scale_factor, x, y)
            self.draw_extras(p, x, y)
    def draw_extras(self, p, x, y): pass
    def _apply_dimming(self, p, mask, x, y):
        p.setBrush(QColor(0, 0, 0, 180)); p.setPen(Qt.PenStyle.NoPen)
        p.save(); p.resetTransform(); p.fillRect(self.rect(), QColor(0,0,0,180)); p.restore()
        try:
            h, w = mask.shape; scale = 1
            if h > 1024 or w > 1024: scale = 2
            work_mask = mask if scale == 1 else cv2.resize(mask, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(work_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            from PyQt6.QtGui import QPainterPath; clip_path = QPainterPath()
            for cnt in contours:
                if scale > 1: cnt = cnt * scale
                pts = [QPointF(x + pt[0][0], y + pt[0][1]) for pt in cnt]
                if len(pts) > 2: clip_path.addPolygon(QPolygonF(pts))
            p.setClipPath(clip_path); p.drawPixmap(QPointF(x, y), self.img_pixmap); p.setClipping(False)
        except Exception: pass 
    def _draw_mask_outline(self, p, mask, color, width, off_x, off_y):
        h, w = mask.shape; scale = 1
        if h > 1024 or w > 1024: scale = 2
        small = mask if scale == 1 else cv2.resize(mask, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pen = QPen(color); pen.setWidthF(width); p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        for cnt in contours:
            if scale > 1: cnt = cnt * scale
            pts = [QPointF(off_x + pt[0][0], off_y + pt[0][1]) for pt in cnt]
            if len(pts) > 2: p.drawPolygon(QPolygonF(pts))
    def wheelEvent(self, e):
        zoom_in = e.angleDelta().y() > 0; mult = 1.1 if zoom_in else 0.9
        m_pos = e.position(); cx, cy = self.width()/2, self.height()/2
        delta = m_pos - QPointF(cx, cy); old_s = self.scale_factor
        new_s = max(0.01, min(50.0, old_s * mult)); factor = new_s / old_s
        self.offset = delta - (delta - self.offset) * factor; self.scale_factor = new_s; self.update()
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True; self.last_mouse_pos = e.position(); self.setCursor(Qt.CursorShape.ClosedHandCursor)
    def mouseMoveEvent(self, e):
        if self.is_dragging:
            delta = e.position() - self.last_mouse_pos; self.offset += delta; self.last_mouse_pos = e.position(); self.update()
        else:
            ix, iy = self.get_img_coords(e.position())
            if self.img_pixmap and 0 <= ix < self.img_pixmap.width() and 0 <= iy < self.img_pixmap.height():
                qimg = self.img_pixmap.toImage(); c = qimg.pixelColor(ix, iy)
                self.pixel_hover.emit((c.blue(), c.green(), c.red()), None)
            else: self.pixel_hover.emit(None, None)
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton: self.is_dragging = False; self.setCursor(Qt.CursorShape.ArrowCursor)

class ReferenceViewer(BaseImageViewer):
    color_picked = pyqtSignal(tuple) 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controls = OverlayControls(self, show_toggle=False)
        self.controls.reset_signal.connect(self.reset_view); self.controls.zoom_100_signal.connect(self.zoom_100)
        self.drag_start_p = None 
    
    def resizeEvent(self, e):
        self.controls.adjustSize(); self.controls.move(self.width() - self.controls.width() - 10, 10)
        self.controls.raise_(); super().resizeEvent(e)
    def mousePressEvent(self, e):
        self.drag_start_p = e.position(); super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.drag_start_p is None: return
        if e.button() == Qt.MouseButton.LeftButton:
            dist = (e.position() - self.drag_start_p).manhattanLength()
            if dist < 3: 
                ix, iy = self.get_img_coords(e.position())
                if self.img_pixmap and 0 <= ix < self.img_pixmap.width() and 0 <= iy < self.img_pixmap.height():
                    c = self.img_pixmap.toImage().pixelColor(ix, iy); self.color_picked.emit((c.blue(), c.green(), c.red()))

class TargetViewer(BaseImageViewer):
    area_selected = pyqtSignal(list); cancel_selection = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cv_original = None; self.cv_processed = None; self.show_result = False
        self.cv_lab_cache = None; self.hover_mask = None; self.selected_masks = []
        self.controls = OverlayControls(self, show_toggle=True)
        self.controls.toggle_signal.connect(self.on_view_toggle)
        self.controls.reset_signal.connect(self.reset_view); self.controls.zoom_100_signal.connect(self.zoom_100)
        self.undo_ctrl = UndoControls(self); self.undo_ctrl.move(10, 10); self.tolerance = 10 
        self.last_valid_coords = None
        self.drag_start_p = None 

    def set_data(self, original, processed):
        self.cv_original = original; self.cv_processed = processed
        if original is not None: self.cv_lab_cache = cv2.cvtColor(original, cv2.COLOR_BGR2Lab)
        else: self.cv_lab_cache = None
        self._refresh_image()
    def on_view_toggle(self, checked): self.show_result = checked; self._refresh_image()
    def _refresh_image(self):
        if self.show_result and self.cv_processed is not None: self.set_image_data(self.cv_processed)
        else: self.set_image_data(self.cv_original)
    
    def set_tolerance(self, val):
        self.tolerance = val

    def resizeEvent(self, e):
        self.controls.adjustSize(); self.controls.move(self.width() - self.controls.width() - 10, 10)
        self.controls.raise_(); super().resizeEvent(e)
    def draw_extras(self, p, off_x, off_y):
        if self.selected_masks:
            for m in self.selected_masks: self._draw_mask_outline(p, m, QColor(255, 255, 0, 255), 2.0/self.scale_factor, off_x, off_y)
    
    def update_hover_mask(self, pos):
        ix, iy = self.get_img_coords(pos)
        self._update_mask_from_coords(ix, iy)

    def _update_mask_from_coords(self, ix, iy):
        c_orig = None; c_res = None
        if self.cv_original is not None and 0 <= ix < self.cv_original.shape[1] and 0 <= iy < self.cv_original.shape[0]:
            c_orig = tuple(self.cv_original[iy, ix])
        if self.cv_processed is not None and 0 <= ix < self.cv_processed.shape[1] and 0 <= iy < self.cv_processed.shape[0]:
            c_res = tuple(self.cv_processed[iy, ix])
        self.pixel_hover.emit(c_orig, c_res)
        
        if self.cv_original is not None and c_orig is not None:
            self.last_valid_coords = (ix, iy)
            
            if self.show_result and self.cv_processed is not None:
                target_color = self.cv_processed[iy, ix]
                lower = np.array([max(0, int(c)-0) for c in target_color], dtype=np.uint8)
                upper = np.array([min(255, int(c)+0) for c in target_color], dtype=np.uint8)
                mask = cv2.inRange(self.cv_processed, lower, upper)
            else:
                target_lab = self.cv_lab_cache[iy, ix]; tol = self.tolerance
                lower = np.array([max(0, int(c)-tol) for c in target_lab], dtype=np.uint8)
                upper = np.array([min(255, int(c)+tol) for c in target_lab], dtype=np.uint8)
                mask = cv2.inRange(self.cv_lab_cache, lower, upper)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            self.hover_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel); self.set_highlight(self.hover_mask)
        else: 
            self.hover_mask = None; self.set_highlight(None)

    def mouseMoveEvent(self, e):
        if self.is_dragging: super().mouseMoveEvent(e); return
        self.update_hover_mask(e.position())
    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.RightButton:
            self.selected_masks = []; self.set_dim_mask(None); self.cancel_selection.emit(); self.update(); return
        self.drag_start_p = e.position(); super().mousePressEvent(e)
    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        if self.drag_start_p is None: return
        if e.button() == Qt.MouseButton.LeftButton:
            dist = (e.position() - self.drag_start_p).manhattanLength()
            if dist < 3: 
                if self.hover_mask is not None:
                    if self.selected_masks:
                        h0, w0 = self.selected_masks[0].shape
                        if self.hover_mask.shape != (h0, w0): self.selected_masks = []
                    self.selected_masks.append(self.hover_mask.copy())
                    if self.selected_masks:
                        h, w = self.selected_masks[0].shape; combined = np.zeros((h, w), dtype=np.uint8)
                        for m in self.selected_masks:
                            if m.shape == (h, w): combined = cv2.bitwise_or(combined, m)
                        self.set_dim_mask(combined); self.area_selected.emit(self.selected_masks)

# ==========================================
# 5. 单张图片处理线程
# ==========================================
class SingleImageWorker(QThread):
    result_ready = pyqtSignal(int, object, object) 
    
    def __init__(self, idx, img, palette, processor, color_space, ui_params):
        super().__init__()
        self.idx = idx
        self.img = img
        self.palette = palette
        self.processor = processor
        self.color_space = color_space
        self.ui_params = ui_params 

    def run(self):
        try:
            res = self.processor.color_transfer(self.img, self.palette, self.color_space)
            self.result_ready.emit(self.idx, res, self.ui_params)
        except Exception as e:
            print(f"Error processing image {self.idx}: {e}")
            self.result_ready.emit(self.idx, None, None)

# ==========================================
# 6. 主窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ColorMapper Pro")
        self.resize(1600, 950)
        
        icon_path = resource_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.ref_image = None; self.images_data = []; self.current_idx = -1
        
        self.processing_queue = []
        self.current_worker = None
        self.processing_active = False
        self.total_batch_count = 0
        self.finished_batch_count = 0
        self.current_batch_params = None 
        
        self.init_ui(); self.apply_theme()
        
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo)
        
        self.color_panel = DraggableColorPanel(self); self.color_panel.show()
        QApplication.processEvents()
        g = self.geometry(); self.color_panel.move(g.x() + g.width()//2 - 80, g.y() + g.height() - 150)

    def init_ui(self):
        main_w = QWidget(); self.setCentralWidget(main_w)
        main_l = QHBoxLayout(main_w); main_l.setContentsMargins(0,0,0,0); main_l.setSpacing(0)

        # === Left Sidebar ===
        side_l = QFrame(); side_l.setFixedWidth(320); side_l.setObjectName("sidebar")
        l_layout = QVBoxLayout(side_l); l_layout.setContentsMargins(10,10,10,10); l_layout.setSpacing(10)
        
        # 文件操作
        io_g = QGroupBox("文件操作"); io_box = QVBoxLayout()
        btn_ref = QPushButton(" 导入色指定图片 (Reference)"); btn_ref.clicked.connect(self.load_ref)
        btn_ref.setToolTip("选择一张图片作为色彩参考，提取其主色调用于后续处理。")
        
        btn_imp = QPushButton(" 导入待处理图片 (Target)"); btn_imp.clicked.connect(self.import_images)
        btn_imp.setToolTip("批量导入需要进行颜色统一的图片。")
        
        btn_exp = QPushButton(" 导出选中图片"); btn_exp.clicked.connect(self.export_images)
        btn_exp.setToolTip("将右侧列表中已勾选的图片导出到指定文件夹。")
        
        io_box.addWidget(btn_ref); io_box.addWidget(btn_imp); io_box.addWidget(btn_exp)
        io_g.setLayout(io_box); l_layout.addWidget(io_g)

        # 自动处理参数
        param_g = QGroupBox("自动处理参数 (Batch)"); p_box = QVBoxLayout(); p_box.setSpacing(5)
        
        # 提取模式选择
        lbl_mode = QLabel("提取模式 (Extraction Mode):")
        lbl_mode.setToolTip("选择从参考图中提取颜色的算法：\n1. K-Means (默认): 计算颜色的数学平均值，适合去噪和概括。\n2. Histogram (直方图): 统计并选取频率最高的 N 个真实颜色，绝不产生新颜色。")
        self.combo_pal_mode = QComboBox()
        self.combo_pal_mode.addItems(["K-Means (聚类平均)", "Histogram (直方图 - 严格匹配)"])
        self.combo_pal_mode.setToolTip(lbl_mode.toolTip())
        p_box.addWidget(lbl_mode)
        p_box.addWidget(self.combo_pal_mode)
        p_box.addSpacing(5)

        # 1. 色彩空间选择
        lbl_space = QLabel("色彩空间 (Color Space):")
        lbl_space.setToolTip("选择用于颜色计算的色彩空间。\nLAB: 最接近人眼感知 (默认/推荐)。\nHSV: 适合调整色相/饱和度。\nRGB: 纯数学距离，不推荐用于色彩匹配。")
        self.combo_space = QComboBox()
        self.combo_space.addItems(["LAB (推荐)", "HSV", "RGB"])
        self.combo_space.setToolTip(lbl_space.toolTip())
        p_box.addWidget(lbl_space)
        p_box.addWidget(self.combo_space)
        p_box.addSpacing(5)

        # 2. 主色调数量 K
        k_tooltip = "调色板大小 (Palette Size)。\nK-Means模式：生成 K 个聚类中心颜色。\nHistogram模式：选取出现频率最高的 N 个颜色。\n数值越大：保留更多色彩细节。\n数值越小：色彩越概括/抽象。"
        self.c_k, self.sl_k, self.sb_k = self.create_slider_input("主色调数量 (N/K):", 2, 64, 16, k_tooltip)
        
        # 添加推荐按钮
        header_layout = self.c_k.layout().itemAt(0).layout() 
        btn_k_rec = QPushButton("推荐"); btn_k_rec.setFixedSize(50, 20)
        btn_k_rec.setStyleSheet("QPushButton { font-size:11px; padding:0px; margin-left: 5px; } QToolTip { font-size: 13px; font-family: \"Microsoft YaHei\"; color: #ffffff; background-color: #333333; border: 1px solid #0078d4; }")
        btn_k_rec.clicked.connect(self.auto_k)
        btn_k_rec.setToolTip("智能分析：\n根据当前选择的模式，自动计算最佳的颜色数量。\nK-Means: 基于肘部法则。\nHistogram: 基于覆盖率 (85%)。")
        header_layout.addWidget(btn_k_rec)
        p_box.addWidget(self.c_k)

        # 3. 魔棒容差
        tol_tooltip = "魔棒容差 (0-100)：\n控制手动选区时的颜色敏感度。\n数值越小：选区越精确，只选中极其相似的颜色。\n数值越大：选区越宽泛，容忍更大的色差。"
        self.c_tol, self.sl_tol, self.sb_tol = self.create_slider_input("魔棒容差 (Tolerance):", 0, 100, 10, tol_tooltip)
        self.sl_tol.valueChanged.connect(self.update_viewer_tolerance) 
        p_box.addWidget(self.c_tol)

        p_box.addSpacing(10)
        self.btn_run = QPushButton("执行处理 (选中图片)"); self.btn_run.setFixedHeight(45)
        self.btn_run.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-weight: bold; border-radius: 5px; font-size: 13px; } QPushButton:hover { background-color: #1084e0; } QPushButton:pressed { background-color: #005a9e; }")
        self.btn_run.clicked.connect(self.toggle_processing)
        self.btn_run.setToolTip("批量处理：\n对右侧列表中【已勾选】的图片执行颜色迁移。\n处理过程中界面不会卡顿，可随时停止。")
        p_box.addWidget(self.btn_run)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        p_box.addWidget(self.progress_bar)

        param_g.setLayout(p_box); l_layout.addWidget(param_g); l_layout.addStretch()
        main_l.addWidget(side_l)

        # === Center View ===
        center_f = QFrame(); c_layout = QVBoxLayout(center_f); c_layout.setContentsMargins(0,0,0,0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.view_ref = ReferenceViewer()
        self.view_ref.pixel_hover.connect(lambda c, _: self.on_hover_sync(c, 'ref'))
        self.view_ref.color_picked.connect(self.on_apply_manual_color)
        self.view_tgt = TargetViewer()
        self.view_tgt.pixel_hover.connect(lambda c1, c2: self.on_hover_sync(c1, 'tgt', c2))
        self.view_tgt.area_selected.connect(self.on_target_selected)
        self.view_tgt.cancel_selection.connect(self.on_target_deselect)
        self.view_tgt.undo_ctrl.undo.connect(self.undo); self.view_tgt.undo_ctrl.redo.connect(self.redo)
        splitter.addWidget(self.view_ref); splitter.addWidget(self.view_tgt)
        splitter.setStretchFactor(0, 1); splitter.setStretchFactor(1, 1)
        c_layout.addWidget(splitter); main_l.addWidget(center_f)

        # === Right Sidebar ===
        side_r = QFrame(); side_r.setFixedWidth(280); side_r.setObjectName("sidebar_right")
        r_layout = QVBoxLayout(side_r); r_layout.setContentsMargins(10,10,10,10)
        l_grp = QGroupBox("图层列表"); lg_l = QVBoxLayout()
        self.table = QTableWidget(); self.table.setColumnCount(2)
        self.table.horizontalHeader().hide(); self.table.verticalHeader().hide(); self.table.setShowGrid(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setColumnWidth(0, 28) 
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.cellClicked.connect(self.on_layer_click); lg_l.addWidget(self.table)
        
        btn_row = QHBoxLayout()
        b_all = QPushButton("全选"); b_all.clicked.connect(lambda: self.sel_layers(True))
        b_all.setToolTip("勾选所有图片")
        b_inv = QPushButton("反选"); b_inv.clicked.connect(self.inv_layers)
        b_inv.setToolTip("反向勾选图片")
        b_del = QPushButton("删除"); b_del.clicked.connect(self.del_layers)
        b_del.setToolTip("从列表中移除已勾选的图片")
        btn_row.addWidget(b_all); btn_row.addWidget(b_inv); btn_row.addWidget(b_del); lg_l.addLayout(btn_row)
        
        l_grp.setLayout(lg_l); r_layout.addWidget(l_grp); main_l.addWidget(side_r)

    def create_slider_input(self, label_text, min_val, max_val, default_val, tooltip=""):
        container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 5, 0, 5); layout.setSpacing(2)
        
        # Label Row
        lbl_layout = QHBoxLayout(); lbl = QLabel(label_text)
        if tooltip: lbl.setToolTip(tooltip); container.setToolTip(tooltip)
        lbl_layout.addWidget(lbl)
        
        # SpinBox
        inp = QSpinBox(); inp.setRange(min_val, max_val); inp.setValue(default_val); inp.setFixedWidth(60); inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if tooltip: inp.setToolTip(tooltip)

        # Slider Row (Slider + Spinbox)
        input_layout = QHBoxLayout()
        slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(min_val, max_val); slider.setValue(default_val)
        if tooltip: slider.setToolTip(tooltip)
        
        slider.valueChanged.connect(inp.setValue); inp.valueChanged.connect(slider.setValue)
        
        input_layout.addWidget(slider); input_layout.addWidget(inp)
        layout.addLayout(lbl_layout); layout.addLayout(input_layout)
        
        return container, slider, inp

    def apply_theme(self):
        BLUE_ACCENT = "#0078d4"
        BLUE_HOVER = "#1084e0"
        BLUE_PRESSED = "#005a9e"
        DARK_BG = "#202020"
        PANEL_BG = "#333333"
        BORDER_COL = "#555555"
        TEXT_COL = "#e0e0e0"
        
        css = f"""
        QMainWindow, QWidget {{ 
            background-color: {DARK_BG}; 
            color: {TEXT_COL}; 
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif; 
            selection-background-color: {BLUE_ACCENT};
            selection-color: white;
        }}
        #sidebar, #sidebar_right {{ background-color: #2b2b2b; border: 1px solid #1f1f1f; }}
        QGroupBox {{ border: 1px solid #444; margin-top: 8px; padding-top: 10px; font-weight: bold; border-radius: 4px; }}
        QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; background-color: #2b2b2b; }}
        
        QTableWidget {{ background-color: {PANEL_BG}; border: 1px solid {BORDER_COL}; color: #eee; outline: none; }}
        QTableWidget::item {{ padding: 2px; border-bottom: 1px solid #3a3a3a; }}
        QTableWidget::item:selected {{ background-color: #444; border: 1px solid {BLUE_ACCENT}; }}
        
        QPushButton {{ background-color: #3a3a3a; border: 1px solid {BORDER_COL}; padding: 5px; border-radius: 4px; }}
        QPushButton:hover {{ background-color: #454545; border-color: #0078d4; }}
        QPushButton:checked {{ background-color: #0078d4; color: white; }}
        QPushButton:pressed {{ background-color: {BLUE_PRESSED}; border-color: {BLUE_PRESSED}; color: white; }}
        
        QSlider::groove:horizontal {{ height: 4px; background: #555; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: #0078d4; width: 14px; margin: -5px 0; border-radius: 7px; }}
        QSlider::handle:horizontal:hover {{ background: {BLUE_HOVER}; }}
        
        QCheckBox {{ spacing: 0px; margin-left: 0px; }}
        QCheckBox::indicator {{ width: 16px; height: 16px; background: #333; border: 1px solid #666; border-radius: 3px; }}
        QCheckBox::indicator:hover {{ border-color: #888; background: #444; }}
        QCheckBox::indicator:checked {{ background: #0078d4; image: {ICON_CHECK_WHITE}; }}
        
        QComboBox {{ background-color: {PANEL_BG}; border: 1px solid {BORDER_COL}; color: #eee; padding: 4px; border-radius: 4px; }}
        QComboBox:focus {{ border: 1px solid {BLUE_ACCENT}; }}

        QSpinBox {{ background: {PANEL_BG}; border: 1px solid {BORDER_COL}; border-radius: 4px; padding: 2px; color: #eee; }}
        QSpinBox:focus {{ border: 1px solid {BLUE_ACCENT}; }}
        
        QSpinBox::up-button {{ 
            subcontrol-origin: border; subcontrol-position: top right; width: 16px; 
            border-left: 1px solid {BORDER_COL}; border-bottom: 1px solid {BORDER_COL}; 
            background-color: #3a3a3a; border-top-right-radius: 4px; margin-top: 1px; margin-right: 1px; 
        }}
        QSpinBox::down-button {{ 
            subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; 
            border-left: 1px solid {BORDER_COL}; border-top: 0px solid {BORDER_COL}; 
            background-color: #3a3a3a; border-bottom-right-radius: 4px; margin-bottom: 1px; margin-right: 1px; 
        }}
        QSpinBox::up-arrow {{ image: {ICON_UP_PATH}; width: 10px; height: 10px; }}
        QSpinBox::down-arrow {{ image: {ICON_DOWN_PATH}; width: 10px; height: 10px; }}
        
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ background-color: #444; border-left: 1px solid {BLUE_HOVER}; }}
        QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{ background-color: {BLUE_ACCENT}; }}
        
        QProgressBar {{ border: 1px solid {BORDER_COL}; border-radius: 4px; text-align: center; color: white; background-color: {PANEL_BG}; }}
        QProgressBar::chunk {{ background-color: {BLUE_ACCENT}; width: 1px; }}

        QToolTip {{ font-size: 13px; font-family: "Microsoft YaHei"; color: #ffffff; background-color: {PANEL_BG}; border: 1px solid {BLUE_ACCENT}; padding: 5px; border-radius: 3px; }}
        """
        self.setStyleSheet(css)

    def update_viewer_tolerance(self, val):
        self.view_tgt.set_tolerance(val)

    def load_ref(self):
        f, _ = QFileDialog.getOpenFileName(self, "选择色指定图片", "", "Images (*.png *.jpg *.webp)")
        if f:
            img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self.ref_image = img; self.view_ref.set_image_data(img)
                self.auto_k() 

    def auto_k(self):
        if self.ref_image is None: QMessageBox.warning(self, "提示", "请先导入色指定图片"); return
        
        mode = self.combo_pal_mode.currentIndex()
        if mode == 0: # K-Means
            k = PROCESSOR.recommend_k(self.ref_image)
        else: # Histogram
            k = PROCESSOR.recommend_unique_n(self.ref_image)
            
        self.sb_k.setValue(k)

    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "导入待处理图片", "", "Images (*.png *.jpg *.webp)")
        for f in files:
            img = cv2.imdecode(np.fromfile(f, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None: continue
            data = {'name': os.path.basename(f), 'original': img, 'processed': img.copy(), 'history': [(img.copy(), None)], 'hist_ptr': 0}
            self.images_data.append(data); self._add_row(data)
        if self.current_idx == -1 and self.images_data: self.table.selectRow(0); self.on_layer_click(0, 0)

    def _add_row(self, data):
        r = self.table.rowCount(); self.table.insertRow(r); self.table.setRowHeight(r, 40)
        
        cw = QWidget(); hl = QHBoxLayout(cw); hl.setContentsMargins(0,0,0,0); hl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        chk = QCheckBox(); chk.setChecked(True); 
        chk.setStyleSheet("margin-left:0; margin-right:0;")
        hl.addWidget(chk); self.table.setCellWidget(r, 0, cw)
        
        item = QTableWidgetItem(data['name'])
        thumb = cv2.resize(data['original'], (32,32)); thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        im = QImage(thumb.data, 32, 32, 3*32, QImage.Format.Format_RGB888)
        item.setIcon(QIcon(QPixmap.fromImage(im))); self.table.setItem(r, 1, item)

    def on_layer_click(self, r, c):
        self.current_idx = r
        self.view_tgt.selected_masks = []; self.view_tgt.hover_mask = None
        self.view_tgt.set_dim_mask(None); self.view_tgt.cancel_selection.emit()
        self._refresh_target_view()

    def _refresh_target_view(self):
        if self.current_idx < 0 or self.current_idx >= len(self.images_data): self.view_tgt.set_data(None, None); return
        d = self.images_data[self.current_idx]; self.view_tgt.set_data(d['original'], d['processed']); self._update_undo_btns()

    def sel_layers(self, val):
        for i in range(self.table.rowCount()): self.table.cellWidget(i,0).findChild(QCheckBox).setChecked(val)
    def inv_layers(self):
        for i in range(self.table.rowCount()): c = self.table.cellWidget(i,0).findChild(QCheckBox); c.setChecked(not c.isChecked())
    def del_layers(self):
        rem = []
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i,0).findChild(QCheckBox).isChecked(): rem.append(i)
        for i in sorted(rem, reverse=True): self.table.removeRow(i); del self.images_data[i]
        if not self.images_data: self.current_idx = -1
        elif self.current_idx >= len(self.images_data): self.current_idx = len(self.images_data)-1
        self._refresh_target_view()

    def on_hover_sync(self, c1, source, c2=None):
        if source == 'tgt':
            self.color_panel.update_colors(c1, c2)
            color_to_match = c2 if (self.view_tgt.show_result and c2 is not None) else c1
            if self.ref_image is not None and color_to_match is not None:
                tol = 10
                lower = np.array([max(0, int(c)-tol) for c in color_to_match], dtype=np.uint8)
                upper = np.array([min(255, int(c)+tol) for c in color_to_match], dtype=np.uint8)
                mask = cv2.inRange(self.ref_image, lower, upper)
                self.view_ref.set_highlight(mask)

    def on_target_selected(self, masks): pass
    def on_target_deselect(self): pass

    def on_apply_manual_color(self, bgr):
        if self.current_idx == -1: return
        masks = self.view_tgt.selected_masks
        if not masks: return
        d = self.images_data[self.current_idx]; curr_state = d['processed'].copy()
        fill_color = np.array(bgr, dtype=np.uint8)
        h_img, w_img = curr_state.shape[:2]; h, w = masks[0].shape; final_mask = np.zeros((h,w), dtype=np.uint8)
        for m in masks:
            if m.shape == (h, w): final_mask = cv2.bitwise_or(final_mask, m)
        if final_mask.shape != (h_img, w_img): return
        curr_state[final_mask > 0] = fill_color
        
        self._push_history(d, curr_state, None)
        d['processed'] = curr_state
        self.view_tgt.selected_masks = []; self.view_tgt.set_dim_mask(None)
        
        if not self.view_tgt.controls.btn_toggle.isChecked():
            self.view_tgt.controls.set_toggle_state(True)
            self.view_tgt.show_result = True 
        
        self._refresh_target_view()

    def _push_history(self, data, new_img, params):
        data['history'] = data['history'][:data['hist_ptr']+1]
        data['history'].append((new_img, params))
        if len(data['history']) > 20: data['history'].pop(0)
        data['hist_ptr'] = len(data['history']) - 1; self._update_undo_btns()

    def undo(self):
        if self.current_idx == -1: return
        d = self.images_data[self.current_idx]
        if d['hist_ptr'] > 0: 
            d['hist_ptr'] -= 1
            img, params = d['history'][d['hist_ptr']]
            d['processed'] = img.copy()
            if params is not None:
                self.sb_k.setValue(params['k'])
                self.combo_pal_mode.setCurrentIndex(params['mode'])
                self.combo_space.setCurrentIndex(params['space'])
            self._refresh_target_view()

    def redo(self):
        if self.current_idx == -1: return
        d = self.images_data[self.current_idx]
        if d['hist_ptr'] < len(d['history']) - 1: 
            d['hist_ptr'] += 1
            img, params = d['history'][d['hist_ptr']]
            d['processed'] = img.copy()
            if params is not None:
                self.sb_k.setValue(params['k'])
                self.combo_pal_mode.setCurrentIndex(params['mode'])
                self.combo_space.setCurrentIndex(params['space'])
            self._refresh_target_view()

    def _update_undo_btns(self):
        if self.current_idx == -1: return
        d = self.images_data[self.current_idx]
        self.view_tgt.undo_ctrl.update_states(d['hist_ptr'] > 0, d['hist_ptr'] < len(d['history']) - 1)

    # ==========================================
    # 队列式批量处理逻辑 (线程安全修正版)
    # ==========================================
    def toggle_processing(self):
        if self.processing_active:
            self.processing_active = False
            self.processing_queue.clear()
            self.btn_run.setText("执行处理 (选中图片)")
            self.btn_run.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-weight: bold; border-radius: 5px; font-size: 13px; } QPushButton:hover { background-color: #1084e0; } QPushButton:pressed { background-color: #005a9e; }")
            self.progress_bar.setVisible(False)
            return
        
        self.run_process_queue()

    def run_process_queue(self):
        if self.ref_image is None: QMessageBox.warning(self, "错误", "需要先导入色指定图片"); return
        
        indices = []
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i,0).findChild(QCheckBox).isChecked(): indices.append(i)
        
        if not indices: 
            QMessageBox.warning(self, "提示", "请在右侧图层列表中至少勾选一张图片。")
            return
        
        self.processing_queue = indices
        self.total_batch_count = len(indices)
        self.finished_batch_count = 0
        self.processing_active = True
        
        self.btn_run.setText(f"停止执行 ({0}/{self.total_batch_count})")
        self.btn_run.setStyleSheet("QPushButton { background-color: #e81123; color: white; font-weight: bold; border-radius: 5px; font-size: 13px; } QPushButton:hover { background-color: #ff4d4d; }")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        k = self.sb_k.value()
        mode = self.combo_pal_mode.currentIndex()
        self.current_batch_params = {
            'k': k,
            'mode': mode,
            'space': self.combo_space.currentIndex()
        }
        
        try:
            if mode == 0:
                self.worker_palette = PROCESSOR.get_dominant_colors(self.ref_image, k)
            else:
                self.worker_palette = PROCESSOR.get_top_unique_colors(self.ref_image, k)
        except Exception as e:
            QMessageBox.critical(self, "提取失败", f"无法提取颜色: {e}")
            self.toggle_processing() # Stop
            return
            
        self.worker_space = self.combo_space.currentText().split()[0]
        self.process_next_image()

    def process_next_image(self):
        # 如果队列为空或手动停止
        if not self.processing_active or not self.processing_queue:
            self.processing_active = False
            self.btn_run.setText("执行处理 (选中图片)")
            self.btn_run.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-weight: bold; border-radius: 5px; font-size: 13px; } QPushButton:hover { background-color: #1084e0; } QPushButton:pressed { background-color: #005a9e; }")
            self.progress_bar.setVisible(False)
            
            # [关键修复] 显式断开引用，防止下次启动时复用已销毁的 C++ 对象
            if self.current_worker is not None:
                # 尝试断开信号连接（如果存在）
                try: self.current_worker.result_ready.disconnect() 
                except: pass
                
                # 强制退出线程并等待
                self.current_worker.quit()
                self.current_worker.wait()
                self.current_worker = None
            
            # [额外安全] 清理 GPU 缓存，防止显存泄漏
            if HAS_CUPY:
                try: cp.get_default_memory_pool().free_all_blocks()
                except: pass

            if self.finished_batch_count == self.total_batch_count:
                print(f"处理完成：共 {self.finished_batch_count} 张图片。")
            return

        idx = self.processing_queue.pop(0)
        img = self.images_data[idx]['original']
        
        # 创建新的 Worker
        self.current_worker = SingleImageWorker(idx, img, self.worker_palette, PROCESSOR, self.worker_space, self.current_batch_params)
        self.current_worker.result_ready.connect(self.on_worker_finished)
        self.current_worker.finished.connect(self.current_worker.deleteLater)
        self.current_worker.start()

    def on_worker_finished(self, idx, res, params):
        if idx < len(self.images_data) and res is not None:
            d = self.images_data[idx]
            self._push_history(d, res, params)
            d['processed'] = res
            if idx == self.current_idx:
                self._refresh_target_view()
        
        self.finished_batch_count += 1
        pct = int(self.finished_batch_count / self.total_batch_count * 100)
        self.progress_bar.setValue(pct)
        self.btn_run.setText(f"停止执行 ({self.finished_batch_count}/{self.total_batch_count})")
        
        # [关键修复] 使用 QTimer.singleShot 异步调用下一张处理
        # 这确保当前线程有足够时间完成清理和退出，避免 "Destroyed while running"
        QTimer.singleShot(0, self.process_next_image)

    def export_images(self):
        has_checked = False
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i, 0).findChild(QCheckBox).isChecked():
                has_checked = True
                break
        
        if not has_checked:
            QMessageBox.warning(self, "提示", "未选择任何图片。\n请在右侧列表中勾选需要导出的图片。")
            return

        d = QFileDialog.getExistingDirectory(self, "选择导出目录"); 
        if not d: return
        cnt = 0
        for i in range(self.table.rowCount()):
            if self.table.cellWidget(i,0).findChild(QCheckBox).isChecked():
                data = self.images_data[i]
                original_basename = os.path.splitext(data['name'])[0]
                target_name = f"{original_basename}_output.png"
                target_path = os.path.join(d, target_name)
                dup_count = 1
                while os.path.exists(target_path):
                    target_name = f"{original_basename}_output ({dup_count}).png"
                    target_path = os.path.join(d, target_name)
                    dup_count += 1
                try:
                    cv2.imencode(".png", data['processed'])[1].tofile(target_path)
                    cnt += 1
                except Exception as e:
                    print(f"Error saving {target_path}: {e}")

        QMessageBox.information(self, "完成", f"已导出 {cnt} 张")

    def closeEvent(self, event):
        RES_MANAGER.cleanup()
        super().closeEvent(event)

if __name__ == '__main__':
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    f = app.font()
    f.setFamily("Microsoft YaHei"); f.setPointSize(9); app.setFont(f)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())