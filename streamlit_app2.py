import streamlit as st
import cv2
import numpy as np
from rtds import FinalSearcher  # replace with actual import path

# Initialize the searcher once and cache it
def init_searcher(api_key: str, conf_thresh: float = 0.5, distance_thresh: float = 0.5):
    return FinalSearcher(api_key=api_key), conf_thresh, distance_thresh

@st.cache_resource
def get_searcher(api_key: str, conf_thresh: float, distance_thresh: float):
    return init_searcher(api_key, conf_thresh, distance_thresh)

# Retrieve API key and thresholds
API_KEY = "AIzaSyC7oRC1pH6T05ZspKqJOVh2P6ARiw0mOqI"
CONF_THRESH = st.sidebar.slider("YOLO Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
DISTANCE_THRESH = st.sidebar.slider("Embedding Distance Threshold", 0.01, 1.0, 0.35, 0.01)

searcher, CONF_THRESH, DISTANCE_THRESH = get_searcher(API_KEY, CONF_THRESH, DISTANCE_THRESH)

st.title("Real-Time Person Embedding & Similarity Search")

# UI controls
metadata_input = st.text_input("Label for current person embedding:")
add_btn = st.button("Store Current Person")
clear_btn = st.button("Clear Embeddings Database")

# Placeholder for video frames and status messages
frame_placeholder = st.empty()
status = st.empty()

# Attempt to open the webcam
cap = cv2.VideoCapture('mrbeast.mp4')  # Replace with 0 for webcam or video file path
if not cap.isOpened():
    st.error("Cannot open camera.")
else:
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                status.error("Failed to read frame.")
                break

            # Handle clear database request
            if clear_btn:
                # Delete all embeddings in ChromaDB
                try:
                    searcher.embedder.collection.delete(where={})
                    status.success("Cleared all embeddings from database.")
                except Exception as e:
                    status.error(f"Failed to clear database: {e}")
                clear_btn = False

            annotated = frame.copy()
            bboxes = searcher.detector.detect(frame)

            for (x1, y1, x2, y2, conf) in bboxes:
                if conf < CONF_THRESH:
                    continue

                # Draw detection box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"Conf: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Query for similar embeddings
                crop = frame[y1:y2, x1:x2]
                if crop.size:
                    metadatas, distances = searcher.embedder.query_similar_images(crop, n_results=1)
                    if distances and distances[0] and distances[0][0] < DISTANCE_THRESH:
                        meta = metadatas[0][0]
                        dist = distances[0][0]
                        # Overlay match box
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f"Match: {meta} ({dist:.2f})"
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated, (x1, y1 - h - 15), (x1 + w, y1), (255, 0, 0), -1)
                        cv2.putText(annotated, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Display the annotated frame
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

            # Handle storing new embeddings
            if add_btn:
                if not metadata_input:
                    status.warning("Enter a label before storing.")
                elif bboxes:
                    x, y, meta, idx = searcher.interrupt(frame, f"Interrupt:{metadata_input}")
                    status.success(f"Stored embedding id={idx}, metadata={meta}")
                else:
                    status.warning("No detected person to store.")
                add_btn = False

            if st.session_state.get("stop_loop", False):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
