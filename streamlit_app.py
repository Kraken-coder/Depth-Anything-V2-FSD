import streamlit as st
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

# --- Detector Class (import from existing module or redefine) ---
class RealTimeSteeringDetector:
    def __init__(self, model_encoder='vits', model_path_template='depth_anything_v2_{}.pth',
                 roi_top_frac=0.6, roi_bottom_frac=1.0, roi_x_start=0,
                 sensor_width=36, sensor_height=24, focal_length=50):
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
        elif torch.backends.mps.is_available():
            self.DEVICE = 'mps'
        else:
            self.DEVICE = 'cpu'
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if model_encoder not in self.model_configs:
            raise ValueError("Unsupported encoder. Choose one of: " + ", ".join(self.model_configs.keys()))
        self.encoder = model_encoder
        self.model = DepthAnythingV2(**self.model_configs[self.encoder])
        model_path = model_path_template.format(self.encoder)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.DEVICE).eval()
        self.roi_top_frac = roi_top_frac
        self.roi_bottom_frac = roi_bottom_frac
        self.roi_x_start = roi_x_start
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.focal_length = focal_length
        self.horizontal_fov = self.calculate_fov(self.sensor_width, self.focal_length)

    def calculate_fov(self, sensor_size, focal_length):
        half_fov = np.arctan(sensor_size / (2 * focal_length))
        return 2 * half_fov * (180.0 / np.pi)

    def process_frame(self, frame):
        h = frame.shape[0]
        top_y = int(h * self.roi_top_frac)
        bottom_y = int(h * self.roi_bottom_frac)
        cropped = frame[top_y:bottom_y, self.roi_x_start:]
        depth = self.model.infer_image(cropped)
        depth_np = np.array(depth).astype(np.float64)
        nor_roi = depth_np.sum(axis=0) / (bottom_y - top_y)
        inversion = -nor_roi + np.max(nor_roi)
        col_center = len(inversion) // 2
        safest = np.array([inversion.argmax(), inversion.max()])
        dest_idx = min(300, len(inversion)-1)
        dest = np.array([dest_idx, inversion[dest_idx]])
        traj = (safest - np.array([col_center,0]) + dest - np.array([col_center,0])) / 2
        steering = (self.horizontal_fov / len(inversion)) * traj[0]
        return steering, depth_np, inversion, top_y, bottom_y

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Real-Time Depth & Steering Visualization")

# Sidebar configuration
st.sidebar.header("Settings")
model_enc = st.sidebar.selectbox("Model Encoder", ['vits','vitb','vitl','vitg'], index=0)
roi_top = st.sidebar.slider("ROI Top Fraction", 0.0, 1.0, 0.6)
roi_bottom = st.sidebar.slider("ROI Bottom Fraction", 0.0, 1.0, 1.0)
sensor_w = st.sidebar.number_input("Sensor Width (mm)", value=36)
sensor_h = st.sidebar.number_input("Sensor Height (mm)", value=24)
focal_len = st.sidebar.number_input("Focal Length (mm)", value=50)

# Initialize detector
@st.cache_resource
def get_detector():
    return RealTimeSteeringDetector(
        roi_top_frac=0.3,
        roi_bottom_frac=0.7,

    )
    
detector = get_detector()

# Video capture
cap = cv2.VideoCapture('http://100.96.200.190:8080/video')

# Create static layout placeholders
col1, col2, col3 = st.columns(3)
frame_slot = col1.empty()
depth_slot = col2.empty()
profile_slot = col3.empty()
metric_slot = st.empty()

stop_button = st.button("Stop Streaming")

# Streaming loop
while cap.isOpened() and not stop_button:
    ret, frame = cap.read()
    if not ret:
        st.error("Cannot read from camera")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    steering, depth_map, profile, top_y, bottom_y = detector.process_frame(frame_rgb)

    # Prepare visuals
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    overlay = frame_rgb.copy()
    cv2.line(overlay, (0, top_y), (overlay.shape[1], top_y), (255,0,0),2)
    cv2.line(overlay, (0, bottom_y), (overlay.shape[1], bottom_y), (255,0,0),2)
    cv2.putText(overlay, f"Angle: {steering:.2f}°", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

    # Update placeholders
    frame_slot.image(overlay, caption="Camera Feed", use_column_width=True)
    depth_slot.image(depth_color, caption="Depth Map (Plasma)", use_column_width=True)
    fig, ax = plt.subplots()
    ax.plot(profile)
    ax.set_title("Free-Space Profile")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Inverted Cost")
    profile_slot.pyplot(fig)
    metric_slot.metric("Steering Angle", f"{steering:.2f}°")

# Cleanup
cap.release()
cv2.destroyAllWindows()
