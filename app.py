import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
from PIL import Image

st.set_page_config(page_title="KMeans Segmentation", layout="wide")

st.title("Image Segmentation using K-Means")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image with OpenCV from uploaded file
    image = np.array(Image.open(uploaded_file))
    if len(image.shape) == 2:  # grayscale
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, c = img.shape

    # Two options: 5 clusters or 10 clusters
    option = st.radio(
        "Select segmentation type:",
        [
            "1: 5 clusters (XY + BGR)",
            "2: 10 clusters (XY + BGR)"
        ]
    )

    if option == "1: 5 clusters (XY + BGR)":
        n_clusters = 5
    else:  # "2: 10 clusters (XY + BGR)"
        n_clusters = 10

    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    # Features with XY
    features_with_xy = np.concatenate(
        [img.reshape((-1, 3)), X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

    st.write(f"Running KMeans with **{n_clusters} clusters**...")

    # 1. Segmentation with XY
    kmeans_with_xy = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(features_with_xy)
    labels_with_xy = kmeans_with_xy.labels_.reshape((h, w))
    segmented_img_with_xy = labels_with_xy.astype('uint8') * int(255/(n_clusters-1))

    # Always show cluster centres for XY
    st.subheader("Cluster centers (B,G,R,X,Y):")
    st.write(kmeans_with_xy.cluster_centers_)

    # 2. Segmentation with only BGR
    features_bgr_only = img.reshape((-1, 3))
    kmeans_bgr_only = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(features_bgr_only)
    labels_bgr_only = kmeans_bgr_only.labels_.reshape((h, w))
    segmented_img_bgr_only = labels_bgr_only.astype('uint8') * int(255/(n_clusters-1))

    # Always show cluster centres for BGR
    st.subheader("Cluster centers (B,G,R):")
    st.write(kmeans_bgr_only.cluster_centers_)

    st.subheader("Results:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")

    with col2:
        st.image(segmented_img_with_xy, caption=f"Segmented with XY ({n_clusters} clusters)")

    with col3:
        st.image(segmented_img_bgr_only, caption=f"Segmented with BGR only ({n_clusters} clusters)")

    # Download buttons
    st.download_button(
        label="Download XY segmentation",
        data=cv2.imencode('.png', segmented_img_with_xy)[1].tobytes(),
        file_name=f"segmented_xy_{n_clusters}clusters.png",
        mime="image/png"
    )
    st.download_button(
        label="Download BGR segmentation",
        data=cv2.imencode('.png', segmented_img_bgr_only)[1].tobytes(),
        file_name=f"segmented_bgr_{n_clusters}clusters.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to start segmentation.")
