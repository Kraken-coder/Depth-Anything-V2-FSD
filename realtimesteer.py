import cv2
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from depth_anything_v2.dpt import DepthAnythingV2

class RealTimeSteeringDetector:
    def __init__(self,
                 model_encoder='vits',
                 model_path_template='depth_anything_v2_{}.pth',
                 # Instead of fixed pixel indices, use normalized ROI factors.
                 roi_top_frac=0.6,   # start ROI at 60% of frame height.
                 roi_bottom_frac=1.0,  # use until the bottom (100%) of frame height.
                 roi_x_start=0,        # x coordinate start remains as is (in pixels)
                 sensor_width=36,    # in mm, example value
                 sensor_height=24,   # in mm, example value
                 focal_length=50):   # in mm, example value
        # Determine device
        if torch.cuda.is_available():
            self.DEVICE = 'cuda'
        elif torch.backends.mps.is_available():
            self.DEVICE = 'mps'
        else:
            self.DEVICE = 'cpu'
        
        # Model configurations for various encoders
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        if model_encoder not in self.model_configs:
            raise ValueError("Unsupported encoder. Choose one of: " + ", ".join(self.model_configs.keys()))
        
        # Load the depth model
        self.encoder = model_encoder
        self.model = DepthAnythingV2(**self.model_configs[self.encoder])
        model_path = model_path_template.format(self.encoder)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model = self.model.to(self.DEVICE).eval()
        
        # Instead of fixed pixel values for ROI, we store fractions.
        self.roi_top_frac = roi_top_frac  
        self.roi_bottom_frac = roi_bottom_frac  
        self.roi_x_start = roi_x_start   # x coordinate start (can be 0 if full width is used)
        
        # Sensor parameters for Field-of-View (FOV) calculations
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.focal_length = focal_length
        
        # Calculate horizontal FOV (in degrees)
        self.horizontal_fov = self.calculate_fov(self.sensor_width, self.focal_length)
    
    def calculate_fov(self, sensor_size, focal_length):
        """
        Calculate the field of view (in degrees) given the sensor size (mm) and focal length (mm).
        """
        half_fov = np.arctan(sensor_size / (2 * focal_length))
        fov = 2 * half_fov * (180.0 / np.pi)
        return fov

    def process_frame(self, frame):
        """
        Process a single camera frame to compute a steering angle based on depth.
        The ROI is computed from the current frame size using normalized factors.
        """
        frame_height = frame.shape[0]
        # Compute the scaled ROI boundaries using normalized fractions.
        top_y = int(frame_height * self.roi_top_frac)
        bottom_y = int(frame_height * self.roi_bottom_frac)
        
        # Crop the frame for the ROI (usually lower part contains the drivable area)
        cropped_image = frame[top_y:bottom_y, self.roi_x_start:]
        
        # Run the depth model to estimate depth. The model expects a NumPy image.
        # It returns a depth map that we display using the plasma colormap.
        depth = self.model.infer_image(cropped_image)
        
        # Convert depth to a numpy array (if not already)
        depth_np = np.array(depth).astype(np.float64)
        
        # Compute a 1D free-space profile by summing along the vertical axis of the ROI.
        roi_height = bottom_y - top_y
        nor_roi = depth_np.sum(axis=0) / roi_height
        
        # Invert the cost: lower values indicate more free space; here we invert so that higher values correspond to less cost.
        inversion = (-1 * nor_roi) + np.max(nor_roi)
        
        # Compute vectors corresponding to the free-space direction for analysis.
        vec_safest = np.array([inversion.argmax(), inversion.max()])
        current_vec = np.array([len(inversion) // 2, 0])
        dest_index = min(300, len(inversion)-1)  # ensure within array bounds
        destination = np.array([dest_index, inversion[dest_index]])
        
        # Compute vectors from the center of the ROI to both the safest and destination points.
        destination_vec = destination - current_vec
        safest_vec = vec_safest - current_vec
        
        # Simple average of these vectors gives a trajectory vector.
        trajec = (safest_vec + destination_vec) / 2
        
        # Determine a candidate column (offset) for steering.
        ou = int(trajec[0] + len(inversion) // 2)
        out = np.array([trajec[0], inversion[ou]])
        
        # Map the column position to a steering angle.
        angle_per_column = self.horizontal_fov / len(inversion)
        steering_angle = angle_per_column * (out[0])
        relative_dist = out[1]
        
        # Return the computed steering angle (in degrees), along with the free-space profile and output vector.
        return steering_angle, inversion, relative_dist, top_y, bottom_y

    def run_realtime(self, camera_index=0, resize_dim=(320, 240), sample_interval=3):
        """
        Runs the real-time processing loop using a camera feed.
        
        - Preprocesses each frame by resizing to a lower quality.
        - Processes one frame every 'sample_interval' seconds.
        - Draws horizontal lines at the top and bottom boundaries of the ROI.
        - Logs the scaling factors from the original capture resolution to the resized frame.
        
        Press 'q' to quit.
        
        :param camera_index: The device index or IP camera URL.
        :param resize_dim: Tuple to resize frame (width, height).
        :param sample_interval: Time in seconds to wait between processing frames.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Error: Unable to open camera")
            return

        # Get original capture resolution.
        orig_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        orig_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # Optionally, set the capture resolution to a desired value.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resize_dim[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resize_dim[1])
        
        # Calculate scaling factors for debugging (if needed).
        scale_factor_x = resize_dim[0] / orig_width if orig_width > 0 else 1
        scale_factor_y = resize_dim[1] / orig_height if orig_height > 0 else 1
        print(f"Scaling factors -- X: {scale_factor_x:.2f}, Y: {scale_factor_y:.2f}")
        
        last_sample_time = 0  # initialize timer
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from camera")
                break
            
            # Always show the live feed and overlay ROI information.
            frame = cv2.resize(frame, resize_dim)
            current_time = time.time()
            
            if current_time - last_sample_time >= sample_interval:
                # Process the frame and compute the steering angle.
                steering_angle, free_space_profile, out_vector, top_y, bottom_y = self.process_frame(frame)
                
                # Update the sample time.
                last_sample_time = current_time
                
                # Overlay the steering angle information.
                display_frame = frame.copy()
                cv2.putText(display_frame, 
                            f"Steering Angle: {steering_angle:.2f} deg, free space: {out_vector:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0), 
                            2)
                
                # Draw horizontal lines indicating the ROI boundaries.
                cv2.line(display_frame, (0, top_y), (display_frame.shape[1], top_y), (255, 0, 0), 2)
                cv2.line(display_frame, (0, bottom_y), (display_frame.shape[1], bottom_y), (255, 0, 0), 2)
                
                # Display the free-space profile using matplotlib.
                plt.close()
                plt.figure(figsize=(5, 3))
                plt.plot(free_space_profile)
                plt.title("Free-space Profile")
                plt.xlabel("Image Column")
                plt.ylabel("Inverted Cost")
                plt.pause(0.001)
                
                cv2.imshow("Camera Feed", display_frame)
            else:
                # Even if not processing a new frame, show the live feed with ROI boundaries.
                display_frame = frame.copy()
                frame_height = frame.shape[0]
                top_y_display = int(frame_height * self.roi_top_frac)
                bottom_y_display = int(frame_height * self.roi_bottom_frac)
                cv2.line(display_frame, (0, top_y_display), (display_frame.shape[1], top_y_display), (255, 0, 0), 2)
                cv2.line(display_frame, (0, bottom_y_display), (display_frame.shape[1], bottom_y_display), (255, 0, 0), 2)
                cv2.putText(display_frame, 
                            f"Steering Angle: {steering_angle:.2f} deg, free space: {out_vector:.2f}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, 
                            (0, 255, 0), 
                            2)
                cv2.imshow("Camera Feed", display_frame)
            
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Example of how to create and run the detector:
if __name__ == "__main__":
    # The ROI will cover from 60% to 100% of the frame height.
    detector = RealTimeSteeringDetector(roi_top_frac=0.3, roi_bottom_frac=0.7)
    # Run the real-time processing loop.
    detector.run_realtime(camera_index='http://100.119.10.212:8080/video', resize_dim=(320, 240), sample_interval=3)
