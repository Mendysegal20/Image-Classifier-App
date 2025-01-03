import streamlit as st
import threading
import os
import signal
import torch
from PIL import Image
from networkx.algorithms.bipartite.basic import color
from torchvision import transforms
from CIFAR10_Model import ColorImgCNN


classes = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck" ]

emojis = {
    "Plane": "✈️",
    "Car": "🚗",
    "Bird": "🐦",
    "Cat": "🐱",
    "Deer": "🦌",
    "Dog": "🐶",
    "Frog": "🐸",
    "Horse": "🐴",
    "Ship": "🚢",
    "Truck": "🚛"
}


def close_server():
    os.kill(os.getpid(), signal.SIGTERM)

if st.button("❌ Close App"):
    close_server()


st.markdown("<h1 style='text-align: center; color: #e3bf52;'>Image Recognizer</h1>", unsafe_allow_html=True)
user_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])


# load the model
model_path = os.path.join(os.path.dirname(__file__), "CNN_CIFAR-10_model.pt")
new_model = ColorImgCNN()
new_model.load_state_dict(torch.load(model_path))


# define the transformation of the image
new_transform = transforms.Compose([
    transforms.Resize((32, 32)), # reduce the image to 32x32 pixels
    transforms.ToTensor(), # make the image a big tensor for the model
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


if user_img is not None:
    img = Image.open(user_img)
    img = new_transform(img)
    img = img.unsqueeze(0) # put the image inside a batch

    new_model.eval()
    with torch.no_grad():
        y_pred = new_model(img)
        prediction = torch.max(y_pred, 1)[1]
        st.markdown(f'<p style="color: #71d62d; font-size: 24px;">The prediction is: {classes[prediction]}</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="font-size: 50px; margin-top: -10px; line-height: 1;">{emojis[classes[prediction]]}</p>',
            unsafe_allow_html=True
        )

    st.markdown("<h3>Was I right?</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        if st.button("YES"):
            st.success("That's why i'm the GOAT, THE GOAT!! 😊")

    with col2:
        if st.button("NO"):
            st.error("That's your fault 🙁")