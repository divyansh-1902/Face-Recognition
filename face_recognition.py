import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from facenet_pytorch import MTCNN
from scipy.spatial.distance import cosine
from collections import defaultdict

# Initialize model
class FaceRecognitionModel(torch.nn.Module):
    def __init__(self):
        super(FaceRecognitionModel, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 128)

    def forward(self, x):
        return self.resnet(x)

# Device and model initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FaceRecognitionModel().to(device)
model.eval()
mtcnn = MTCNN(keep_all=True, device=device)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Known faces database
known_faces = defaultdict(lambda: None)  # Dictionary to store known face embeddings

def add_face_to_db(name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: The file {image_path} does not exist or cannot be opened.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(img_tensor).cpu().numpy().flatten()

    known_faces[name] = embedding

def recognize_face(embedding, known_faces):
    min_distance = float('inf')
    recognized_name = "Unknown"

    embedding = embedding.flatten()

    for name, known_embedding in known_faces.items():
        if known_embedding is not None:
            known_embedding = known_embedding.flatten()
            distance = cosine(embedding, known_embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_name = name

    return recognized_name

def recognize_from_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is None:
            print("No faces detected.")
            continue

        print("Faces detected:", len(boxes))  # Add this line to see if faces are detected

        faces = [img_rgb[int(box[1]):int(box[3]), int(box[0]):int(box[2])] for box in boxes]

        face_tensors = [preprocess(Image.fromarray(face)).unsqueeze(0).to(device) for face in faces]

        with torch.no_grad():
            embeddings = model(torch.cat(face_tensors)).cpu().numpy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            embedding = embeddings[i]
            recognized_name = recognize_face(embedding, known_faces)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
add_face_to_db('Person_1', "C:/Users/ASUS/Desktop/Projects/face recognition model/20231112_205300.jpg")
recognize_from_webcam()
  