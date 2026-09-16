from ultralytics import settings, YOLO

# Set datasets directory
settings.update({"datasets_dir":   "/Users/bennettpotter/Desktop/Coding/AI-Birds/Bird-AI-ID/datasets/train"})

# Load model
model = YOLO("/Users/bennettpotter/Desktop/Coding/AI-Birds/Bird-AI-ID/best.pt")

# Train the model
model.train(
    epochs=1,
    imgsz=256,
    device="mps",
    verbose=False,
    data="/Users/bennettpotter/Desktop/Coding/AI-Birds/Bird-AI-ID/datasets/nz_birds_dataset_split",
    patience=15
)
results = model.val(
    imgsz=256,
    device="mps",
    verbose=False,
    data="/Users/bennettpotter/Desktop/Coding/AI-Birds/Bird-AI-ID/datasets/nz_birds_dataset_split",)

for result in results:
    boxes = result.boxes      # None for classification
    masks = result.masks      # None for classification
    keypoints = result.keypoints  # None for classification
    probs = result.probs      # Probs object containing classification predictions
    obb = result.obb          # None for classification
    
    result.show()             # Display prediction on screen
    result.save(filename="result.jpg")  # Save prediction output to disk

# Process and display inference results
for result in results:
    boxes = result.boxes      # None for classification
    masks = result.masks      # None for classification
    keypoints = result.keypoints  # None for classification
    probs = result.probs      # Probs object containing classification predictions
    obb = result.obb          # None for classification
    
    result.show()             # Display prediction on screen
    result.save(filename="result.jpg")  # Save prediction output to disk

# Export the model
success = model.export(format="onnx")
