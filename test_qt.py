
import sys
import socket
import struct
import numpy as np
import cv2
import time
import queue
import copy

import matplotlib
import os
os.environ["QT_API"] = "pyqt6"
matplotlib.use('qtagg')

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from enum import IntEnum

PROTOCOL_MAGIC = 0x5AA5
PROTOCOL_VERSION = 0x01
HEADER_FMT = "<HBBHIQBB"   # magic, version, type, len, seq, timestamp_us, flags, reserved
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class PacketType(IntEnum):
    IMU_SEGMENT   = 0x01   # 单个 IMU：segment_id + quat
    FINGERS       = 0x02   # 手指：hand_id + 5 float
    GRIPPER       = 0x03   # 夹爪：opening + yaw
    COMBINED_ALL  = 0x10   # 全部 IMU：多个 segment_id + quat
    VIDEO_FRAME   = 0x30   # 视频帧：cam_id + jpeg


def calc_checksum(data: bytes) -> int:
    """ 16-bit sum → 取反 """
    s = sum(data) & 0xFFFF
    return (~s) & 0xFFFF


def verify_packet(packet: bytes):
    """
    校验并解析整个数据包:
    返回 (header_dict, payload) 或 (None, None)
    """
    if len(packet) < HEADER_SIZE + 2:
        return None, None

    # 解析头
    magic, ver, pkt_type, plen, seq, ts_us, flags, reserved = struct.unpack(
        HEADER_FMT, packet[:HEADER_SIZE]
    )

    if magic != PROTOCOL_MAGIC or ver != PROTOCOL_VERSION:
        return None, None

    expected = HEADER_SIZE + plen + 2
    if len(packet) < expected:
        return None, None

    received_crc = struct.unpack("<H", packet[HEADER_SIZE + plen: HEADER_SIZE + plen + 2])[0]
    calc = calc_checksum(packet[:HEADER_SIZE + plen])
    if received_crc != calc:
        return None, None

    header = {
        "type": PacketType(pkt_type),
        "length": plen,
        "seq": seq,
        "timestamp_us": ts_us,
    }

    payload = packet[HEADER_SIZE: HEADER_SIZE + plen]
    return header, payload


# =========================
#   上半身运动学
# =========================
class BodyKinematics:
    def __init__(self):
        self.LEN_HEAD = 2.5
        self.LEN_SHOULDER = 2.0
        self.LEN_UPPER = 2.5
        self.LEN_FORE = 2.2
        self.LEN_HAND = 1.0

        self.VEC_HEAD = np.array([0, 0, 1])
        self.VEC_L_ARM = np.array([-1, 0, 0])
        self.VEC_R_ARM = np.array([1, 0, 0])

    def quat_to_matrix(self, q):
        w, x, y, z = q
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z),     2*(x*z+w*y)],
            [2*(x*y+w*z),     1-2*(x**2+z**2), 2*(y*z-x*w)],
            [2*(x*z-w*y),     2*(y*z+x*w),     1-2*(x**2+y**2)]
        ])

    def compute_points(self, rotations):
        points = {}
        neck_pos = np.array([0.0, 0.0, 0.0])
        points['neck'] = neck_pos
        points['l_shoulder'] = neck_pos + np.array([-self.LEN_SHOULDER, 0, -0.5])
        points['r_shoulder'] = neck_pos + np.array([self.LEN_SHOULDER, 0, -0.5])

        R_head = rotations.get('head', np.eye(3))
        points['head'] = neck_pos + R_head @ (self.VEC_HEAD * self.LEN_HEAD)

        R_lu = rotations.get('l_upper', np.eye(3))
        points['l_elbow'] = points['l_shoulder'] + R_lu @ (self.VEC_L_ARM * self.LEN_UPPER)
        R_lf = rotations.get('l_fore', np.eye(3))
        points['l_wrist'] = points['l_elbow'] + R_lf @ (self.VEC_L_ARM * self.LEN_FORE)
        R_lh = rotations.get('l_hand', np.eye(3))
        points['l_palm'] = points['l_wrist'] + R_lh @ (self.VEC_L_ARM * self.LEN_HAND)

        R_ru = rotations.get('r_upper', np.eye(3))
        points['r_elbow'] = points['r_shoulder'] + R_ru @ (self.VEC_R_ARM * self.LEN_UPPER)
        R_rf = rotations.get('r_fore', np.eye(3))
        points['r_wrist'] = points['r_elbow'] + R_rf @ (self.VEC_R_ARM * self.LEN_FORE)
        R_rh = rotations.get('r_hand', np.eye(3))
        points['r_palm'] = points['r_wrist'] + R_rh @ (self.VEC_R_ARM * self.LEN_HAND)

        return points


# =========================
#   手指条形图小控件
# =========================
class FingerGauge(QtWidgets.QWidget):
    def __init__(self, name, color="#0f0"):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)

        self.lbl = QtWidgets.QLabel(name)
        self.lbl.setFixedWidth(40)
        self.lbl.setStyleSheet("color: #aaa; font-size: 10px;")

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(
            "QProgressBar { background: #222; border: 0px; border-radius: 3px; }"
            f"QProgressBar::chunk {{ background: {color}; border-radius: 3px; }}"
        )

        layout.addWidget(self.lbl)
        layout.addWidget(self.bar)

    def set_val(self, val):
        self.bar.setValue(int(max(0.0, min(1.0, val)) * 100))


class HandsStatusWidget(QtWidgets.QFrame):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background: #1e1e1e; border-top: 1px solid #444; border-bottom: 1px solid #444;")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)

        l_grp = QtWidgets.QGroupBox("Left Hand Fingers")
        l_grp.setStyleSheet("color: #e67e22; font-weight: bold; border: 0px;")
        l_layout = QtWidgets.QVBoxLayout(l_grp)
        self.l_gauges = []
        for name in ["Thumb", "Index", "Mid", "Ring", "Pinky"]:
            g = FingerGauge(name, "#f97909")
            l_layout.addWidget(g)
            self.l_gauges.append(g)

        r_grp = QtWidgets.QGroupBox("Right Hand Fingers")
        r_grp.setStyleSheet("color: #1abc9c; font-weight: bold; border: 0px;")
        r_layout = QtWidgets.QVBoxLayout(r_grp)
        self.r_gauges = []
        for name in ["Thumb", "Index", "Mid", "Ring", "Pinky"]:
            g = FingerGauge(name, "#1abc9c")
            r_layout.addWidget(g)
            self.r_gauges.append(g)

        layout.addWidget(l_grp)
        layout.addWidget(r_grp)

    def update_fingers(self, l_data, r_data):
        for i, val in enumerate(l_data[:5]):
            self.l_gauges[i].set_val(val)
        for i, val in enumerate(r_data[:5]):
            self.r_gauges[i].set_val(val)


# =========================
#   上半身 3D
# =========================
class UpperBody3DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 5), tight_layout=True)
        self.figure.patch.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#1e1e1e')

    def _config_ax(self):
        self.ax.clear()
        self.ax.axis('off')

        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')

        self.ax.set_xlim(-5, 5)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-5, 5)

        self.ax.view_init(elev=20, azim=45)

        axis_len = 6.0
        self.ax.plot([0, axis_len], [0, 0], [0, 0], color='red', linewidth=2)
        self.ax.text(axis_len, 0, 0, "X", color='red')
        self.ax.plot([0, 0], [0, axis_len], [0, 0], color='green', linewidth=2)
        self.ax.text(0, axis_len, 0, "Y", color='green')
        self.ax.plot([0, 0], [0, 0], [0, axis_len], color='blue', linewidth=2)
        self.ax.text(0, 0, axis_len, "Z", color='blue')

    def plot_seg(self, p1, p2, c, lw=3):
        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=c, linewidth=lw)
        self.ax.scatter(p1[0], p1[1], p1[2], color='white', s=25)

    def update_pose(self, pts):
        self._config_ax()
        self.plot_seg(pts['l_shoulder'], pts['r_shoulder'], '#666')
        self.plot_seg((pts['l_shoulder'] + pts['r_shoulder']) / 2, pts['neck'], '#666')

        self.plot_seg(pts['neck'], pts['head'], '#f1c40f', 4)

        self.plot_seg(pts['l_shoulder'], pts['l_elbow'], '#e74c3c')
        self.plot_seg(pts['l_elbow'], pts['l_wrist'], '#e67e22')
        self.plot_seg(pts['l_wrist'], pts['l_palm'], '#d35400')

        self.plot_seg(pts['r_shoulder'], pts['r_elbow'], '#3498db')
        self.plot_seg(pts['r_elbow'], pts['r_wrist'], '#2980b9')
        self.plot_seg(pts['r_wrist'], pts['r_palm'], '#1abc9c')

        self.canvas.draw_idle()


# =========================
#   手指 3D
# =========================
class Finger3DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111, projection='3d')
        self.l_data = [0] * 5
        self.r_data = [0] * 5

    def _config_ax(self):
        self.ax.clear()
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except:
            pass
        self.ax.axis('off')
        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')

        axis_len = 3.0
        self.ax.plot([0, axis_len], [0, 0], [0, 0], color='red', linewidth=2)
        self.ax.text(axis_len, 0, 0, "X", color='red')
        self.ax.plot([0, 0], [0, axis_len], [0, 0], color='green', linewidth=2)
        self.ax.text(0, axis_len, 0, "Y", color='green')
        self.ax.plot([0, 0], [0, 0], [0, axis_len], color='blue', linewidth=2)
        self.ax.text(0, 0, axis_len, "Z", color='blue')

        self.ax.set_xlim(-7, 7)
        self.ax.set_ylim(-5, 5)
        self.ax.set_zlim(-1, 4)
        self.ax.view_init(elev=20, azim=45)

    def update_fingers(self, l_fingers, r_fingers):
        self.l_data = list(l_fingers[:5])
        self.r_data = list(r_fingers[:5])
        self._draw()

    def _draw_finger_chain(self, base, flex_val, is_thumb=False, color='#e67e22', is_left=True):
        if is_thumb:
            seg_lens = [0.8, 0.6]
            max_angles = [50, 60]
        else:
            seg_lens = [1.0, 0.8, 0.6]
            max_angles = [40, 70, 50]

        angles = [max(0.0, min(1.0, flex_val)) * a for a in max_angles]

        pts = [base.copy()]
        cur = base.copy()
        cum_angle = 0.0

        for L, ang in zip(seg_lens, angles):
            cum_angle += ang
            rad = np.deg2rad(cum_angle)
            if is_left:
                dx = -L * np.sin(rad)
                dy = 0
                dz = L * np.cos(rad)
            else:
                dx = 0
                dy = -L * np.sin(rad)
                dz = L * np.cos(rad)

            cur = cur + np.array([dx, dy, dz])
            pts.append(cur.copy())

        pts = np.array(pts)
        self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=3)
        self.ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color='white', s=5)

    def _draw_hand(self, center, data, color, is_left=True):
        cx, cy, cz = center
        spacing = 1.3

        if is_left:
            offsets = [-(i + 1) * spacing for i in range(5)]
            bases = [np.array([cx, cy + o, cz]) for o in offsets]
        else:
            offsets = [-(i + 1) * spacing for i in range(5)]
            bases = [np.array([cx + o, cy, cz]) for o in offsets]
        
        for i, flex in enumerate(data):
            self._draw_finger_chain(
                base=bases[i],
                flex_val=flex,
                is_thumb=(i == 0),
                color=color,
                is_left=is_left
            )

        palm_w = 1.2
        palm_h = 0.8

        if is_left:
            px = [cx] * 5
            py = [cy - palm_w / 2, cy + palm_w / 2, cy + palm_w / 2,
                  cy - palm_w / 2, cy - palm_w / 2]
        else:
            px = [cx - palm_w / 2, cx + palm_w / 2, cx + palm_w / 2,
                  cx - palm_w / 2, cx - palm_w / 2]
            py = [cy] * 5

        pz = [cz - palm_h / 2] * 5
        self.ax.plot(px, py, pz, color=color, linewidth=1)

    def _draw(self):
        self._config_ax()
        self._draw_hand(center=np.array([3.0, 0.0, 0.0]), data=self.l_data,
                        color='#e67e22', is_left=True)
        self._draw_hand(center=np.array([0.0, 3.0, 0.0]), data=self.r_data,
                        color='#1abc9c', is_left=False)
        self.canvas.draw_idle()


# =========================
#   夹爪 3D
# =========================
class Gripper3DWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111, projection='3d')
        self.ax.set_facecolor('none')

        # normalized opening value (conceptually 0.0..1.0, but we'll allow limiting)
        self.opening = 0.5
        # allowed normalized opening range (values are in same normalized units)
        self.min_opening = 0.0
        self.max_opening = 1.0

        # mapping to a physical gap value used when drawing (meters/units arbitrary)
        self.min_gap = 0.1
        self.max_gap = 1.1

        # yaw rotation in degrees
        self.rotation = 0.0

    def set_opening_limits(self, min_open: float, max_open: float):
        """Set allowed normalized opening range (values in [0, 1]).
        If min_open > max_open the values are swapped internally.
        """
        try:
            mn = float(min_open)
            mx = float(max_open)
        except Exception:
            return
        if mn > mx:
            mn, mx = mx, mn
        # clamp to sane [0,1] bounds
        self.min_opening = max(0.0, mn)
        self.max_opening = min(1.0, mx)

    def set_gap_limits(self, min_gap: float, max_gap: float):
        """Set physical gap range used for drawing the gripper.
        Values can be arbitrary positive numbers (min_gap < max_gap enforced).
        """
        try:
            mn = float(min_gap)
            mx = float(max_gap)
        except Exception:
            return
        if mx <= mn:
            # ensure at least a tiny positive span
            mx = mn + 1e-3
        self.min_gap = mn
        self.max_gap = mx

    def _config_ax(self):
        self.ax.clear()
        self.ax.axis('off')
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except:
            pass
        self.ax.set_facecolor('#1e1e1e')
        self.figure.patch.set_facecolor('#1e1e1e')
        self.ax.view_init(elev=20, azim=45)

        axis_len = 1.0
        self.ax.plot([0, axis_len], [0, 0], [0, 0], color='red', linewidth=2)
        self.ax.text(axis_len, 0, 0, "X", color='red')
        self.ax.plot([0, 0], [0, axis_len], [0, 0], color='green', linewidth=2)
        self.ax.text(0, axis_len, 0, "Y", color='green')
        self.ax.plot([0, 0], [0, 0], [0, axis_len], color='blue', linewidth=2)
        self.ax.text(0, 0, axis_len, "Z", color='blue')

        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.set_zlim(-1, 2)

    def update_gripper(self, opening, yaw_deg):
        # clamp incoming opening to allowed normalized range
        try:
            o = float(opening)
        except Exception:
            o = self.opening

        # clamp to configured [min_opening, max_opening]
        if o < self.min_opening or o > self.max_opening:
            # silently clamp (could log or show a warning if desired)
            o = max(self.min_opening, min(self.max_opening, o))

        # store as normalized opening (keeps the same 0..1 semantic)
        self.opening = o

        # clamp rotation to reasonable range (-180..180)
        try:
            r = float(yaw_deg)
        except Exception:
            r = self.rotation
        if r <= -180.0:
            r = -180.0
        elif r >= 180.0:
            r = 180.0
        self.rotation = r
        self._draw()

    def _draw(self):
        self._config_ax()

        # map normalized opening (which might be in [min_opening, max_opening])
        # to a physical gap between min_gap and max_gap
        span = max(1e-6, (self.max_opening - self.min_opening))
        frac = (self.opening - self.min_opening) / span
        frac = max(0.0, min(1.0, frac))
        gap = self.min_gap + frac * (self.max_gap - self.min_gap)
        L = 2.5

        yaw = np.deg2rad(self.rotation)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(yaw), -np.sin(yaw)],
            [0, np.sin(yaw), np.cos(yaw)]
        ])

        left = np.array([0, +gap / 2, 0.0])
        right = np.array([0, -gap / 2, 0.0])
        left2 = left + np.array([L, 0, 0])
        right2 = right + np.array([L, 0, 0])

        left, left2 = Rx @ left, Rx @ left2
        right, right2 = Rx @ right, Rx @ right2

        self.ax.plot([left[0], left2[0]], [left[1], left2[1]], [left[2], left2[2]],
                     color="#3498db", linewidth=6)
        self.ax.plot([right[0], right2[0]], [right[1], right2[1]], [right[2], right2[2]],
                     color="#3498db", linewidth=6)

        mid1 = Rx @ np.array([0, +gap / 2, 0.0])
        mid2 = Rx @ np.array([0, -gap / 2, 0.0])
        self.ax.plot(
            [mid1[0], mid2[0]],
            [mid1[1], mid2[1]],
            [mid1[2], mid2[2]],
            color="#5dade2",
            linewidth=6
        )

        self.canvas.draw_idle()


# =========================
#   UDP Worker / Processor
# =========================

class UDPVideoWorker(QtCore.QThread):
    """ 视频：接收完整协议包，类型=VIDEO_FRAME """
    def __init__(self, host, port, q, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.q = q
        self._running = False
        self.sock = None

    def run(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.sock.bind((self.host, self.port))
        except Exception as e:
            print("VideoWorker bind error:", e)
            self._running = False

        while self._running:
            try:
                packet, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except:
                if self._running:
                    print("VideoWorker recv error")
                break

            header, payload = verify_packet(packet)
            if header is None:
                continue
            if header["type"] != PacketType.VIDEO_FRAME:
                continue

            if len(payload) < 4:
                continue

            cam_id = payload[0]
            jpeg = payload[4:]   # 跳过 1 byte cam_id + 3 reserved

            try:
                self.q.put_nowait((cam_id, jpeg))
            except queue.Full:
                pass

        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None

    def stop(self):
        self._running = False
        if self.sock:
            try: self.sock.close()
            except: pass
        self.wait(1000)

class VideoProcessor(QtCore.QThread):
    """
    独立线程：从队列取视频包，解码为 QImage，维护最新帧字典，
    按最大 30 FPS 频率发出快照到 UI
    """
    frames_updated = QtCore.pyqtSignal(object)  # dict{cam_id: QImage}

    def __init__(self, packet_queue, parent=None):
        super().__init__(parent)
        self.packet_queue = packet_queue
        self._running = False
        self.latest_frames = {}  # cam_id -> QImage
        self.max_fps = 30.0

    def run(self):
        self._running = True
        last_emit = time.perf_counter()

        while self._running:
            try:
                cam_id, jpeg = self.packet_queue.get(timeout=0.02)
            except queue.Empty:
                # 检查是否需要发送一次（防止长时间无包时 UI 不刷新）
                now = time.perf_counter()
                if (now - last_emit) > (1.0 / self.max_fps) and self.latest_frames:
                    snap = {cid: img.copy() for cid, img in self.latest_frames.items()}
                    self.frames_updated.emit(snap)
                    last_emit = now
                continue

            # 解码
            arr = np.frombuffer(jpeg, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            q = QtGui.QImage(rgb.data, w, h, c * w, QtGui.QImage.Format.Format_RGB888)
            self.latest_frames[cam_id] = q.copy()

            # 限帧发送
            now = time.perf_counter()
            if (now - last_emit) >= (1.0 / self.max_fps):
                snap = {cid: img.copy() for cid, img in self.latest_frames.items()}
                self.frames_updated.emit(snap)
                last_emit = now

    def stop(self):
        self._running = False
        self.wait(1000)


class UDPDataWorker(QtCore.QThread):
    """接收数据包（IMU/手指/夹爪），不解析，推入队列"""
    def __init__(self, host, port, q, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.q = q
        self._running = False
        self.sock = None

    def run(self):
        self._running = True
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.5)
        try:
            self.sock.bind((self.host, self.port))
        except Exception as e:
            print("DataWorker bind error:", e)
            self._running = False

        while self._running:
            try:
                data, _ = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except:
                if self._running:
                    print("DataWorker recv error")
                break

            if not data:
                continue

            try:
                self.q.put_nowait(data)
            except queue.Full:
                pass

        if self.sock:
            try: self.sock.close()
            except: pass
            self.sock = None

    def stop(self):
        self._running = False
        if self.sock:
            try: self.sock.close()
            except: pass
        self.wait(1000)


class DataProcessor(QtCore.QThread):
    state_updated = QtCore.pyqtSignal(object)

    def __init__(self, q, parent=None):
        super().__init__(parent)
        self.q = q
        self._running = False

        self.last_quats = {
            "head": [1,0,0,0],
            "l_upper":[1,0,0,0],
            "l_fore":[1,0,0,0],
            "l_hand":[1,0,0,0],
            "r_upper":[1,0,0,0],
            "r_fore":[1,0,0,0],
            "r_hand":[1,0,0,0],
        }

        self.last_l_f = [0]*5
        self.last_r_f = [0]*5
        self.last_gr = (0.5,0.0)

    def run(self):
        self._running = True
        last_emit = time.perf_counter()

        while self._running:
            try:
                packet = self.q.get(timeout=0.05)
            except queue.Empty:
                now = time.perf_counter()
                if now - last_emit > 0.02:
                    self.emit_state()
                    last_emit = now
                continue

            header, payload = verify_packet(packet)
            if header is None:
                continue

            ptype = header["type"]

            # ---------- IMU ------------
            if ptype == PacketType.IMU_SEGMENT:
                if len(payload) >= 20:
                    seg_id = payload[0]
                    w,x,y,z = struct.unpack("<4f", payload[4:20])
                    name_map = {
                        0:"head",1:"l_upper",2:"l_fore",3:"l_hand",
                        4:"r_upper",5:"r_fore",6:"r_hand"
                    }
                    name = name_map.get(seg_id)
                    if name:
                        self.last_quats[name] = [w,x,y,z]

            # ----------- Fingers ---------
            elif ptype == PacketType.FINGERS:
                if len(payload) >= 24:
                    hand = payload[0]
                    f = struct.unpack("<5f", payload[4:24])
                    if hand == 0:
                        self.last_l_f = list(f)
                    elif hand == 1:
                        self.last_r_f = list(f)

            # ----------- Gripper ---------
            elif ptype == PacketType.GRIPPER:
                if len(payload) >= 8:
                    opening, yaw = struct.unpack("<2f", payload[:8])
                    self.last_gr = (opening, yaw)

            elif ptype == PacketType.COMBINED_ALL:
                if len(payload) >= 160:
                    vals = struct.unpack("<40f", payload[:160])
                    keys = ["head","l_upper","l_fore","l_hand","r_upper","r_fore","r_hand"]
                    for i,k in enumerate(keys):
                        idx = i*4
                        self.last_quats[k] = vals[idx:idx+4]
                    self.last_l_f = list(vals[28:33])
                    self.last_r_f = list(vals[33:38])
                    self.last_gr = (vals[38], vals[39])

            # 限帧更新 UI
            now = time.perf_counter()
            if now - last_emit > 0.02:
                self.emit_state()
                last_emit = now

    def emit_state(self):
        snap = {
            "quats": copy.deepcopy(self.last_quats),
            "l_fingers": list(self.last_l_f),
            "r_fingers": list(self.last_r_f),
            "gripper": tuple(self.last_gr),
        }
        self.state_updated.emit(snap)

    def stop(self):
        self._running = False
        self.wait(1000)

# =========================
#   主窗口
# =========================
class BodyMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RK3588 Full System Monitor")
        self.resize(1600, 950)
        self.setStyleSheet("background-color: #1e1e1e; color: #fff;")

        self.kinematics = BodyKinematics()
        self.last_pose = {}
        self.last_fingers_l = [0]*5
        self.last_fingers_r = [0]*5
        self.last_gripper = (0.5, 0.0)
        self.last_quats = {
            'head': [1, 0, 0, 0],
            'l_upper': [1, 0, 0, 0],
            'l_fore': [1, 0, 0, 0],
            'l_hand': [1, 0, 0, 0],
            'r_upper': [1, 0, 0, 0],
            'r_fore': [1, 0, 0, 0],
            'r_hand': [1, 0, 0, 0],
        }

        # --- 队列 ---
        self.video_queue = queue.Queue(maxsize=100)
        self.data_queue = queue.Queue(maxsize=200)

        # --- 视频 Worker + Processor ---
        self.video_worker = UDPVideoWorker("0.0.0.0", 8889, self.video_queue)
        self.video_proc = VideoProcessor(self.video_queue)
        self.video_proc.frames_updated.connect(self.on_video_snapshot)
        self.video_worker.start()
        self.video_proc.start()

        # --- 数据 Worker + Processor ---
        self.data_worker = UDPDataWorker("0.0.0.0", 8888, self.data_queue)
        self.data_proc = DataProcessor(self.data_queue)
        self.data_proc.state_updated.connect(self.on_data_snapshot)
        self.data_worker.start()
        self.data_proc.start()

        # 最新快照（UI 定时器读取）
        self.latest_frames = {}   # cam_id -> QImage
        self.latest_state = None  # {'quats', 'l_fingers', 'r_fingers', 'gripper'}

        self.init_ui()

        # UI 刷新限帧 30FPS
        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self.update_ui_from_snapshot)
        self.ui_timer.start(33)  # ~30 fps

    # --------- UI 搭建 ----------
    def init_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        main_split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(main_split)

        # 左边：6 路视频
        left_w = QtWidgets.QWidget()
        left_g = QtWidgets.QGridLayout(left_w)
        left_g.setContentsMargins(0, 0, 0, 0)
        self.cams = {}

        class VidW(QtWidgets.QWidget):
            def __init__(self, t):
                super().__init__()
                l = QtWidgets.QVBoxLayout(self)
                l.setContentsMargins(1, 1, 1, 1)
                title = QtWidgets.QLabel(t)
                self.lb = QtWidgets.QLabel()
                self.lb.setStyleSheet("background:#000; border:1px solid #444")
                self.lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.lb.setFixedSize(480, 260)
                l.addWidget(title)
                l.addWidget(self.lb)

            def set_img(self, im: QtGui.QImage):
                if im is None:
                    return
                pix = QtGui.QPixmap.fromImage(im).scaled(
                    self.lb.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.lb.setPixmap(pix)

        configs = [
            (0, "Depth1 Color", 0, 0),
            (1, "Depth1 Depth", 0, 1),
            (2, "RGB Left_1", 1, 0),
            (3, "RGB Right_1", 1, 1),
            (4, "RGB Left_2", 2, 0),
            (5, "RGB Right_2", 2, 1)
        ]

        for i, t, r, c in configs:
            w = VidW(t)
            left_g.addWidget(w, r, c)
            self.cams[i] = w

        for i in range(3):
            left_g.setRowStretch(i, 1)

        main_split.addWidget(left_w)

        # 右边三块 3D
        right_split = QtWidgets.QSplitter(Qt.Orientation.Vertical)

        # 1. 上半身
        top_container = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_container)
        top_layout.addWidget(QtWidgets.QLabel("1. IMU 3D Kinematics (Head + Arms)"))
        self.viz_3d = UpperBody3DWidget()
        top_layout.addWidget(self.viz_3d)
        right_split.addWidget(top_container)

        # 2. 手指
        mid_container = QtWidgets.QWidget()
        mid_layout = QtWidgets.QVBoxLayout(mid_container)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.addWidget(QtWidgets.QLabel("2. Finger 3D Pose (Both Hands)"))
        self.fingers3d = Finger3DWidget()
        mid_layout.addWidget(self.fingers3d)
        right_split.addWidget(mid_container)

        # 3. 夹爪
        bot_container = QtWidgets.QWidget()
        bot_layout = QtWidgets.QVBoxLayout(bot_container)
        bot_layout.addWidget(QtWidgets.QLabel("3. End Effector (3D Gripper)"))
        self.grip3d = Gripper3DWidget()
        self.grip3d.setFixedHeight(260)
        bot_layout.addWidget(self.grip3d)
        right_split.addWidget(bot_container)

        right_split.setStretchFactor(0, 2)
        right_split.setStretchFactor(1, 2)
        right_split.setStretchFactor(2, 4)

        main_split.addWidget(right_split)
        main_split.setSizes([1000, 600])

    # --------- 工作线程 → UI 快照 ----------
    @QtCore.pyqtSlot(object)
    def on_video_snapshot(self, frames_dict):
        # 只保存，不立即画；真正画在 ui_timer 里
        self.latest_frames = frames_dict

    @QtCore.pyqtSlot(object)
    def on_data_snapshot(self, state_dict):
        self.latest_state = state_dict

    # --------- UI 定时刷新 ----------
    def update_ui_from_snapshot(self):

        if self.latest_frames:

            for fixed_id in [0,1,2,3,4,5]:
                img = self.latest_frames.get(fixed_id)
                if img is not None:
                    self.cams[fixed_id].set_img(img)

        # ---------- 3D 状态 ----------
        if self.latest_state is None:
            return

        state = self.latest_state

        # ========== 1. IMU 只更新正常收到的部分 ==========
        if 'quats' in state:
            for k, q in state['quats'].items():
                self.last_quats[k] = q

        mats = {k: self.kinematics.quat_to_matrix(q)
                for k, q in self.last_quats.items()}
        pts = self.kinematics.compute_points(mats)
        self.viz_3d.update_pose(pts)

        # ========== 2. 左手手指（部分更新不闪动） ==========
        if 'l_fingers' in state:
            if len(state['l_fingers']) == 5:
                self.last_fingers_l = state['l_fingers']
        self.fingers3d.update_fingers(self.last_fingers_l, self.last_fingers_r)

        # ========== 3. 右手手指 ==========
        if 'r_fingers' in state:
            if len(state['r_fingers']) == 5:
                self.last_fingers_r = state['r_fingers']
        self.fingers3d.update_fingers(self.last_fingers_l, self.last_fingers_r)

        # ========== 4. 夹爪 ==========
        if 'gripper' in state:
            g = state['gripper']
            if isinstance(g, (list, tuple)) and len(g) == 2:
                self.last_gripper = g

        opening, yaw = self.last_gripper
        self.grip3d.update_gripper(opening, yaw)

    def closeEvent(self, e):
        # 停止线程
        self.video_worker.stop()
        self.video_proc.stop()
        self.data_worker.stop()
        self.data_proc.stop()
        e.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = BodyMonitorWindow()
    win.show()
    sys.exit(app.exec())
