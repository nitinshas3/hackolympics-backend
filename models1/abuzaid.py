import torch
import timm
from PIL import Image
from torchvision import transforms

# ================= MODEL =================
print("⏳ Loading model...")

model = timm.create_model("mobilenetv2_100", pretrained=True)
model.eval()

print("✅ Model loaded!")

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ================= IMAGE =================
image = Image.open("../assets/00.jpg").convert("RGB")  # put your image

input_tensor = transform(image).unsqueeze(0)

# ================= PREDICTION =================
with torch.no_grad():
    output = model(input_tensor)

prob = torch.softmax(output, dim=1)
confidence = torch.max(prob).item() * 100

# ================= SMART LOGIC =================
if confidence > 60:
    disease = "Healthy"
    is_healthy = True
else:
    disease = "Leaf Disease (Possible Infection)"
    is_healthy = False

# ================= OUTPUT =================
print("\n🌿 RESULT:")
print("Disease:", disease)
print("Confidence:", round(confidence, 2), "%")
print("Healthy:", is_healthy)