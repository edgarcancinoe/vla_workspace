from fastapi import FastAPI, Request
import torch
import base64
import cv2
import numpy as np
import uvicorn
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

app = FastAPI()

# 1. Load the model (on GPU)
print("Loading SmolVLA to GPU...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)
policy.eval()

@app.post("/forward")
async def forward(request: Request):
    data = await request.json()
    
    # 2. Decode the compressed image
    img_bytes = base64.b64decode(data["laptop"])
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB and Torch Tensor (expected by most VLA models)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float() / 255.0
    
    # 3. Get joint positions from payload
    # Note: You might need to adjust keys to match what your policy expects
    state = torch.tensor([data.get(k, 0.0) for k in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]]).unsqueeze(0).to(device)

    # 4. Inference
    with torch.no_grad():
        observation = {
            "observation.images.laptop": image_tensor,
            "observation.state": state
        }
        action = policy.select_action(observation)

    # 5. Return action to Mac
    return {"action": action.squeeze(0).tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)