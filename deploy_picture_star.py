# .\empre\Scripts\activate
# .\star_proy\Scripts\activate
from PIL import Image
import pandas as pd
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models, transforms
import torch
from resnet_FT import ResNetGAPFeatures as Net
from dataset_e import AestheticsDataset,create_dataloader
import streamlit as st
torch.cuda.empty_cache()

# Load the model and define other functions

def process_image(image):
    # Perform image processing tasks here
    # For demonstration purposes, let's just resize the image
    resized_image = image.resize((300, 300))
    return resized_image

def convert_path(path):
    return path.replace("\\", "/")

def extract_pooled_features(inp, net):
    _ = net(inp)
    pooled_features = [features.feature_maps for features in net.all_features] 
    return pooled_features

def downsample_pooled_features(features):
    dim_reduced_features = []
    for pooled_feature in features:
        if pooled_feature.size()[-1] == 75:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size=(7, 7)))
        elif pooled_feature.size()[-1] == 38:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (4, 4), padding=1))
        elif pooled_feature.size()[-1] == 19:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (2, 2), padding=1))
        elif pooled_feature.size()[-1] == 10:
            dim_reduced_features.append(pooled_feature)
    dim_reduced_features = torch.cat(dim_reduced_features, dim=1).squeeze()
    return dim_reduced_features


def scale(image, low=-1, high=1):
    im_max = np.max(image)
    im_min = np.min(image)
    return (high - low) * (image - np.min(image))/(im_max - im_min) + low 

def extract_heatmap(features, weights, w, h):
    cam = np.zeros((10, 10), dtype=np.float32) 
    temp = weights.view(-1, 1, 1) * downsampled_pooled_features
    summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
    cam = cam + summed_temp
    cam = cv2.resize(cam, (w, h))
    cam = scale(cam)
    return cam 

def extract_prediction(inp, net,all_keys):
    d = dict()
    net.eval()
    output = net(inp)
    for i, key in enumerate(all_keys):
        d[key] = output[:, i].squeeze().item()
    return d

def estrellas(image_score):
    if image_score<=0:
        resultado='OMG, your image is ugly AF... 🤮'
    elif 0<image_score and image_score<=0.50:
        resultado='here are 2 stars for the effort: ⭐⭐'
    elif 0.50<image_score and image_score<=0.65:
        resultado='this is giving 3 stars: ⭐⭐⭐'
    elif 0.65<image_score and image_score<=0.75:
        resultado='look at u! ur almost a pro: ⭐⭐⭐⭐'
    elif 0.75<image_score:
        resultado='IT IS GI-VING INFLUECER VIBZ: ⭐⭐⭐⭐⭐'
    return resultado

# -------------------------------------------------------------------------------------
# Streamlit web app
def main():
    # Set page config
    st.set_page_config(
        page_title="SnapSter",
        page_icon="📷",
        layout="centered",
        initial_sidebar_state="auto"
    )
        # Title and description
    st.title("SnapSter")
    st.markdown("An app to check how aesthetic is your pic")
    st.write("---")

    st.title("Welcome to SnapSter!!")
    st.write("Wanna know the potential of the pic you just took? Is it gonna be engaging? How to know if my followers are going to love it before posting?")
    st.write("Fear no more! SnapSter is here to help")
    st.write("We'll let you know, in a scale from ⭐ to ⭐⭐⭐⭐⭐ how pretty and engaging your picture is going to be")
    st.write("Try it out! we don't bite....sometimes 👼")
    st.write("---")

    # Instructions
    st.subheader("Instructions")
    st.write("1. Click Browse files")
    st.write("2. Select the pic that you want to check")
    st.write("3. Hit the *Here we go* button")
    st.write("4. Check on a scale from 1 to 5 stars (⭐) how ✨AESTHETIC✨ is your pic!")
    st.write("---")
    # load model
    checkpoint='epoch_7_loss_0_3686805795728361.pth'
    resnet=models.resnet50(weights="IMAGENET1K_V1")
    model = Net(resnet, n_features=12)
    model.load_state_dict(torch.load(f"{checkpoint}", map_location=torch.device('cpu')))
#     model.load_state_dict(torch.load(f"{checkpoint}", map_location=lambda storage, loc: storage))

    # key creation
    attr_keys = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF',
                'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
    non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
    all_keys = attr_keys + non_neg_attr_keys
    used_keys=all_keys
    weights = {k: model.attribute_weights.weight[i, :] for i, k in enumerate(all_keys)} 

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Run button
    if st.button("Here we go") and uploaded_file:
        dataloader_ = create_dataloader(uploaded_file, is_train=False)
        for algo in dataloader_:
            print(algo)


        for imagenes in dataloader_:
            image = imagenes['image']
            predicted_values = extract_prediction(image, model,all_keys)
            # data={'score': predicted_values['score']}
            calif_final=estrellas(predicted_values['score'])

        
        # Display the results
        st.write("Results:")
        st.write(calif_final)
        # st.write(predicted_values['score'])


# Run the app
if __name__ == "__main__":
    main()
