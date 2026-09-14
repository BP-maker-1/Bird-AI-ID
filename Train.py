from ultralytics import settings, YOLO

# Set datasets directory
settings.update({"datasets_dir": "path to data set"})

# Load model
model = YOLO("yolo26n-cls.pt")

# Train the model
model.train(
    epochs=1,
    imgsz=256,
    device="mps",
    verbose=False,
    data="path to data set",
    patience=15
)

# Run inference on images (replace source with your image file or directory path)
results = model.predict(
    source="path to data set",
    imgsz=256,
    device="mps"
)

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
