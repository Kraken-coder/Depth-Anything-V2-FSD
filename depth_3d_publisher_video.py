import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import cv2
import torch
import torch.cuda.amp as amp
import numpy as np
import os
from depth_anything_v2.dpt import DepthAnythingV2

# Enable expandable segments and mixed precision
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def free_gpu_memory():
    """Utility to clear cache and collect garbage."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

class Depth3DPublisher(Node):
    def __init__(self):
        super().__init__('depth_3d_publisher_video')
        self.publisher_ = self.create_publisher(PointCloud2, 'depth_points', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # OpenCV video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            rclpy.shutdown()

        # Determine device and enable mixed precision on CUDA
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'

        # Load depth model
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48,  96,  192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536,1536,1536,1536]}
        }
        encoder = 'vits'
        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(self.device)
        if self.device == 'cuda':
            self.model = self.model.half()
        self.model.eval()

        # Define cropping window (rows, cols)
        self.crop_h = (400, 600)
        self.crop_w = (0, None)

        # Camera intrinsics (replace with real calibration)
        self.fx = 525.0
        self.fy = 525.0

        self.get_logger().info('Depth3DPublisher (video) node initialized.')

    def depth_to_point_cloud(self, depth: np.ndarray) -> np.ndarray:
        """Vectorized conversion from depth map to Nx3 float32 array."""
        h, w = depth.shape
        uu, vv = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.flatten()
        mask = z > 0
        z = z[mask]
        u = uu.flatten()[mask]
        v = vv.flatten()[mask]

        x = (u - (w/2)) * z / self.fx
        y = (v - (h/2)) * z / self.fy
        points = np.stack((x, y, z), axis=-1)
        return points.astype(np.float32)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image from camera")
            return

        y0, y1 = self.crop_h
        x0, x1 = self.crop_w
        cropped = frame[y0:y1, x0:x1]
        if cropped.size == 0:
            self.get_logger().warn("Crop window out of bounds")
            return

        # Inference with no_grad and autocast for mixed precision
        with torch.no_grad():
            if self.device == 'cuda':
                with amp.autocast():
                    depth_map = self.model.infer_image(cropped)
            else:
                depth_map = self.model.infer_image(cropped)

        depth = np.array(depth_map, dtype=np.float32)
        points = self.depth_to_point_cloud(depth)
        if points.size == 0:
            self.get_logger().warn("No valid depth points to publish")
            free_gpu_memory()
            return

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "camera_depth_optical_frame"
        cloud_msg = pc2.create_cloud_xyz32(header, points)
        self.publisher_.publish(cloud_msg)
        self.get_logger().info(f"Published point cloud with {points.shape[0]} points")

        del depth_map, depth, points
        free_gpu_memory()

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Depth3DPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
