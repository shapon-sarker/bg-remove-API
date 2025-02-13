from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
from torchvision import transforms
from model.u2net import U2NET

app = FastAPI()


model_path = "u2net.pth"
try:
    model = U2NET(3, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise e



def remove_background(image: Image.Image) -> Image.Image:
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320)),
        ])
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            prediction = model(input_tensor)[0][:, 0, :, :]  # Correct indexing
            print(f"🔍 Prediction shape: {prediction.shape}")

        # ✅ Tensor -> NumPy কনভার্সন
        mask = prediction.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())  # Normalize
        mask = (mask * 255).astype(np.uint8)  # Convert to uint8

        # ✅ Resize with proper interpolation
        mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

        # ✅ Apply mask to RGBA image
        rgba_image = image.convert("RGBA")
        data = np.array(rgba_image)
        data[..., 3] = mask  # Apply mask to alpha channel
        return Image.fromarray(data)

    except Exception as e:
        print(f"❌ Error in remove_background: {str(e)}")
        raise ValueError(f"Error while removing background: {str(e)}") from e


# ✅ API Endpoint
@app.post("/remove_bg/")
async def remove_bg(file: UploadFile = File(...)):
    try:
        # ✅ ইমেজ লোড চেক
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        print(f"📸 Image loaded successfully: {file.filename}")
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return {"error": f"Failed to load image: {str(e)}"}

    try:
        result = remove_background(image)

        # ✅ ইমেজ PNG হিসেবে রিটার্ন
        img_io = BytesIO()
        result.save(img_io, format='PNG')
        img_io.seek(0)
        return {"filename": file.filename, "message": "✅ Background removed successfully!"}

    except Exception as e:
        print(f"❌ Error in processing image: {str(e)}")
        return {"error": f"Failed to process image: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
