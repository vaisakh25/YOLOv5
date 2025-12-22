import torch
import cv2
from PIL import Image
from torchvision.transforms import transforms
from yolov5.models.experimental import attempt_load

# Load the pre-trained YOLOv5 model
weights = 'yolov5s.pt'  # Path to the model weights
model = attempt_load(weights, device=torch.device('cpu'))

# Set device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
])

# # Load and preprocess the input image
# image_path = 'path/to/your/image.jpg'
# image = Image.open(image_path).convert('RGB')
# image_tensor = transform(image).unsqueeze(0).to(device)

# # Perform inference
# with torch.no_grad():
#     detections = model(image_tensor)

# # Parse the detections
# results = []
# for det in detections.pred:
#     boxes = det[:, :4].detach().cpu().numpy()
#     scores = det[:, 4].detach().cpu().numpy()
#     labels = det[:, 5].detach().cpu().numpy().astype(int)
    
#     for box, score, label in zip(boxes, scores, labels):
#         result = {
#             'box': box.tolist(),
#             'score': score,
#             'label': label
#         }
#         results.append(result)

# Load the object tracker
tracker = cv2.TrackerCSRT_create()

# Read the input video
video_path = 'path/to/your/video.mp4'
video = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not video.isOpened():
    print("Error opening video file")
    exit()

# Read the first frame
success, frame = video.read()
if not success:
    print("Error reading video file")
    exit()

# Initialize the tracker
bbox = cv2.selectROI("YOLOv5 Tracking", frame, fromCenter=False, showCrosshair=True)
tracker.init(frame, bbox)

# Object count
object_count = 0

# Process frames
while True:
    # Read a new frame
    success, frame = video.read()
    if not success:
        break

    # Convert the frame to PIL Image and apply transformations
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        detections = model(frame_tensor)

    # Parse the detections
    results = []
    for det in detections.pred:
        boxes = det[:, :4].detach().cpu().numpy()
        scores = det[:, 4].detach().cpu().numpy()
        labels = det[:, 5].detach().cpu().numpy().astype(int)

        for box, score, label in zip(boxes, scores, labels):
            result = {
                'box': box.tolist(),
                'score': score,
                'label': label
            }
            results.append(result)

    # Update the tracker
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Visualize the results
    for result in results:
        box = result['box']
        score = result['score']
        label = result['label']
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'{score:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.putText(frame, f'{label}', (int(x1), int(y1) - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        # Increment object count
        object_count += 1

    # Display the object count
    cv2.putText(frame, f'Object Count: {object_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow('YOLOv5 Tracking', frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()