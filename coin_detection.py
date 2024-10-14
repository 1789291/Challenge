import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict
import uuid

app = FastAPI()

# Store uploaded images and detected circular object data
IMAGE_STORE = "coin-dataset/"
if not os.path.exists(IMAGE_STORE):
    os.makedirs(IMAGE_STORE)

# Dictionary to store circular object data
circular_objects_db = {}

# Data model for bounding box and object details
class CircularObject(BaseModel):
    id: str
    bounding_box: List[int]
    centroid: List[int]
    radius: float

class CircularObjectListResponse(BaseModel):
    objects: List[CircularObject]

# Endpoint 1: Upload Image and Detect Circular Objects
@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # Save the uploaded image
    file_location = os.path.join(IMAGE_STORE, file.filename)
    with open(file_location, "wb+") as f:
        f.write(file.file.read())

    # Read the image using OpenCV
    img = cv2.imread(file_location, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Detect circular objects using Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=10, maxRadius=100)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        objects = []
        
        for (x, y, r) in circles:
            object_id = str(uuid.uuid4())
            bounding_box = [x-r, y-r, x+r, y+r]
            centroid = [x, y]
            radius = r
            
            # Store the circular object data
            circular_objects_db[object_id] = {
                "bounding_box": bounding_box,
                "centroid": centroid,
                "radius": r
            }
            objects.append({
                "id": object_id,
                "bounding_box": bounding_box,
                "centroid": centroid,
                "radius": r
            })
        
        return CircularObjectListResponse(objects=objects)
    
    return JSONResponse(status_code=400, content={"message": "No circular objects found."})

# Endpoint 2: Retrieve list of circular objects for a queried image
@app.get("/objects/")
async def get_circular_objects():
    object_list = [
        {"id": obj_id, "bounding_box": obj["bounding_box"], "centroid": obj["centroid"], "radius": obj["radius"]}
        for obj_id, obj in circular_objects_db.items()
    ]
    return {"objects": object_list}

# Endpoint 3: Get bounding box, centroid, and radius for a specific object
@app.get("/object/{object_id}")
async def get_object_details(object_id: str):
    if object_id not in circular_objects_db:
        return JSONResponse(status_code=404, content={"message": "Object not found"})
    
    return circular_objects_db[object_id]

# Endpoint 4: Evaluate model (stub for now)
@app.get("/evaluate/")
async def evaluate_model():
    # Placeholder for evaluation strategy (you can expand this as needed)
    return {"message": "Evaluation strategy not yet implemented."}

# Display circular objects on the original image (mask and circles)
@app.get("/display/{image_name}")
async def display_circular_objects(image_name: str):
    image_path = os.path.join(IMAGE_STORE, image_name)
    
    if not os.path.exists(image_path):
        return JSONResponse(status_code=404, content={"message": "Image not found."})

    img = cv2.imread(image_path)
    
    # Draw circles on the image
    for obj_id, obj in circular_objects_db.items():
        x, y = obj["centroid"]
        r = obj["radius"]
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, encoded_image = cv2.imencode('.jpg', img_rgb)
    
    return JSONResponse(status_code=200, content={"image_data": encoded_image.tobytes().hex()})

# Run the app using Uvicorn server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
