import cv2
import json
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import chromadb
from ultralytics import YOLO
from pydantic import BaseModel
from google import genai
from google.genai import types
class RealTimeEmbedder:
    def __init__(self, model_name: str = 'resnet50', storage_dir: str = "memory", device: str = None):
        client = chromadb.PersistentClient(path=storage_dir)
        try:
            client.create_collection(name="image_embeddings")
        except Exception:
            pass
        self.collection = client.get_collection(name="image_embeddings")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = model_name.lower()
        if model_name.startswith('resnet'):
            base = models.resnet50(pretrained=True)
            self.model = torch.nn.Sequential(*list(base.children())[:-1])
        elif model_name.startswith('mobilenet'):
            base = models.mobilenet_v2(pretrained=True)
            self.model = base.features
            self.model.add_module('pool', torch.nn.AdaptiveAvgPool2d((1,1)))
        else:
            raise ValueError("Model not supported. Use 'resnet50' or 'mobilenet_v2'.")

        self.model = self.model.to(self.device).eval()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        img_t = self.preprocess(frame).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(img_t)
        emb = emb.squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb)

    def store_image_embedding(self, frame: np.ndarray, metadata: dict, id: str) -> np.ndarray:
        emb = self.get_embedding(frame)
        self.collection.add(embeddings=[emb], metadatas=[metadata], ids=[id])
        print(f"Stored embedding for {id}: {metadata}")
        return emb

    def query_similar_images(self, frame: np.ndarray, n_results: int = 5):
        emb = self.get_embedding(frame)
        res = self.collection.query(query_embeddings=[emb], n_results=n_results)
        return res['metadatas'], res['distances']

class Coordinates(BaseModel):
    x_coordinate: int
    y_coordinate: int
    metadata: str
    id: str

class LLMHandler:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def get_result(self, frame: np.ndarray, prompt: str) -> dict:
        success, encoded = cv2.imencode('.jpg', frame)
        if not success:
            raise RuntimeError("Failed to encode image")
        img_bytes = encoded.tobytes()
        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'), prompt],
            config={
                'response_mime_type': 'application/json',
                'response_schema': Coordinates,
            }
        )
        return json.loads(response.text)

class YOLOHumanDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'cpu', conf_thresh: float = 0.25):
        self.model = YOLO(model_path)
        self.model.fuse()
        self.device = device
        self.conf_thresh = conf_thresh
        self.person_class_id = 0

    def detect(self, image: np.ndarray):
        results = self.model.predict(source=image, device=self.device, conf=self.conf_thresh, classes=[self.person_class_id], verbose=False)
        bboxes = []
        if results:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                bboxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return bboxes

    def draw_boxes(self, image: np.ndarray, bboxes: list) -> np.ndarray:
        for x1, y1, x2, y2, conf in bboxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'Person: {conf:.2f}'
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return image

class FinalSearcher:
    def __init__(self, api_key: str, storage_dir: str = 'memory'):
        self.embedder = RealTimeEmbedder(model_name='mobilenet_v2', storage_dir=storage_dir)
        self.detector = YOLOHumanDetector(conf_thresh=0.5)
        self.llm = LLMHandler(api_key=api_key)

    def yolo_checker(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video source")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            bboxes = self.detector.detect(frame)
            for x1, y1, x2, y2, conf in bboxes:
                if conf > 0.5:
                    metadata, distances = self.embedder.query_similar_images(frame[y1:y2, x1:x2])
                    if distances[0][0] < 0.5:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = f'Person: {conf:.2f} {metadata[0]}'
                        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), (255, 0, 0), -1)
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            output = self.detector.draw_boxes(frame, bboxes)
            cv2.imshow('Real-time Human Detection', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def interrupt(self, frame: np.ndarray, query: str):
        out = self.llm.get_result(frame, query)
        x, y, metadata, id_val = out['x_coordinate'], out['y_coordinate'], out['metadata'], out['id']
        bboxes = self.detector.detect(frame)
        for x1, y1, x2, y2, conf in bboxes:
            if x1 < x < x2 and y1 < y < y2:
                crop = frame[y1:y2, x1:x2]
                self.embedder.store_image_embedding(crop, {"source": metadata}, id_val)
                break
        print(f"Stored embedding for {id_val}: {metadata}")
        return x, y, metadata, id_val

if __name__ == '__main__':
    API_KEY = ''
    searcher = FinalSearcher(api_key=API_KEY)

    # Example interrupt usage:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        x, y, meta, idx = searcher.interrupt(frame, "Interrupt:person wearing black shirt is diptanshu")
        print(f"Interrupt result: x={x}, y={y}, metadata={meta}, id={idx}")

    # Continuous YOLO check:
    searcher.yolo_checker(video_source=0)
