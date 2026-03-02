import streamlit as st
import numpy as np
import zipfile
import tempfile
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Internal Image Matcher", layout="wide")
st.title("🧠 Internal Image Matching Tool")

MAX_IMAGES = 500

# ===============================
# Load Model Once
# ===============================
@st.cache_resource
def load_model():
    return SentenceTransformer("clip-ViT-B-32")

model = load_model()

# ===============================
# Initialize Session
# ===============================
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.images = []
    st.session_state.ready = False

# ===============================
# Upload ZIP
# ===============================
st.header("1️⃣ Upload ZIP (Max 500 Images)")

zip_file = st.file_uploader("Upload ZIP file containing images", type=["zip"])

if zip_file and not st.session_state.ready:

    with st.spinner("Processing images... Please wait."):

        image_list = []
        embeddings = []

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            file_list = zip_ref.namelist()

            for file_name in file_list:
                if file_name.lower().endswith(("png", "jpg", "jpeg")):
                    if len(image_list) >= MAX_IMAGES:
                        break
                    try:
                        with zip_ref.open(file_name) as file:
                            img = Image.open(file).convert("RGB")
                            image_list.append(img)
                            emb = model.encode(img)
                            embeddings.append(emb)
                    except:
                        continue

        if len(image_list) == 0:
            st.error("No valid images found inside ZIP.")
            st.stop()

        embeddings = np.array(embeddings)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        st.session_state.index = index
        st.session_state.images = image_list
        st.session_state.ready = True

    st.success(f"✅ {len(image_list)} images indexed successfully!")

# ===============================
# Search Section
# ===============================
if st.session_state.ready:

    st.header("2️⃣ Upload Query Image")

    query_file = st.file_uploader("Upload image to search", type=["png", "jpg", "jpeg"])

    if query_file:

        query_img = Image.open(query_file).convert("RGB")
        st.image(query_img, caption="Query Image", width=300)

        query_embedding = model.encode(query_img)
        query_embedding = np.array([query_embedding])

        D, I = st.session_state.index.search(query_embedding, 3)

        st.header("🔍 Top 3 Matches")

        for rank, idx in enumerate(I[0]):
            distance = D[0][rank]
            similarity = 1 / (1 + distance)
            percentage = round(similarity * 100, 2)

            st.subheader(f"#{rank+1} — Match: {percentage}%")
            st.image(st.session_state.images[idx], width=300)

# ===============================
# Reset Button
# ===============================
if st.session_state.ready:
    if st.button("🔄 Reset Session"):
        st.session_state.index = None
        st.session_state.images = []
        st.session_state.ready = False
        st.experimental_rerun()
