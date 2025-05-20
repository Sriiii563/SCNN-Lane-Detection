import torch
from scnn import SCNN
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import os

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    img = cv2.resize(img, (800, 288))  # match training resolution
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img_rgb).unsqueeze(0).cpu()  # moved to CPU
    return img, img_tensor

def infer_and_draw(model, img_path):
    original_img, img_tensor = preprocess_image(img_path)

    with torch.no_grad():
        seg_pred, exist_pred, _, _, _ = model(img_tensor)

    seg_mask = torch.argmax(seg_pred.squeeze(), dim=0).cpu().numpy()

    lane_colors = {
        1: (0, 255, 0),     # Green
        2: (0, 0, 255),     # Red
        3: (255, 0, 0),     # Blue
        4: (0, 255, 255),   # Yellow
    }

    overlay = original_img.copy()
    for lane_id, color in lane_colors.items():
        overlay[seg_mask == lane_id] = color

    blended = cv2.addWeighted(original_img, 0.6, overlay, 0.4, 0)

    # Display result
    #cv2.imshow("Detected Lanes", blended)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Optional: Save result
    cv2.imwrite("test4.jpg", blended)
    print("Saved as test4.jpg")

if __name__ == "__main__":
    print("Starting Lane detection")
    model_path = "load the pth file path"
    image_path = "load the image path"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    print("Loading model")
    model = SCNN(input_size=(800, 288))  # ✅ fix here
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    model.cpu()

    print("Running inference")
    infer_and_draw(model, image_path)
    print("Done")
