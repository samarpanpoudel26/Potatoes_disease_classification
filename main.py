from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load model from repository folder, NOT local computer path
model = tf.keras.models.load_model("model/1.keras")

class_name = ["Early Blight", "Late Blight", "Healthy"]

def read_file(data):
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file(await file.read())
    image_batch = np.expand_dims(image, 0)
    
    predictions = model.predict(image_batch)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1200)
