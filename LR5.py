
import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
import torch.nn.functional as F


device = torch.device("cpu")

model = models.resnet18(pretrained=True)
model.eval()
model.to(device)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("CPU-Based Image Classification App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probs = F.softmax(output, dim=1)
    top_probs, top_idxs = torch.topk(probs, 5)

    labels = pd.read_csv(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        header=None
    )

    results = []
    for i in range(5):
        results.append({
            "Class": labels.iloc[top_idxs[0][i].item()][0],
            "Probability": float(top_probs[0][i])
        })

    df = pd.DataFrame(results)
    st.dataframe(df)
    st.bar_chart(df.set_index("Class"))
