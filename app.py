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
    st.session_state.image_data = []
    st.session_state.ready = False

# ===============================
# Upload ZIP Section
# ===============================
st.header("1️⃣ Upload ZIP (Max 500 Images)")

zip_file = st.file_uploader("Upload ZIP file containing images", type=["zip"])

if zip_file and not st.session_state.ready:

    with st.spinner("Processing images... Please wait."):

        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "images.zip")

            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)

            image_paths = []
            embeddings = []

            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    if file.lower().endswith(("png", "jpg", "jpeg")):
                        if len(image_paths) >= MAX_IMAGES:
                            break
                        path = os.path.join(root, file)
                        try:
                            img = Image.open(path).convert("RGB")
                            emb = model.encode(img)
                            embeddings.append(emb)
                            image_paths.append(path)
                        except:
                            continue

            if len(image_paths) == 0:
                st.error("No valid images found inside ZIP.")
                st.stop()

            embeddings = np.array(embeddings)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            st.session_state.index = index
            st.session_state.image_data = image_paths
            st.session_state.ready = True

    st.success(f"✅ {len(image_paths)} images indexed successfully!")

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

            matched_img = Image.open(st.session_state.image_data[idx])
            st.image(matched_img, width=300)

# ===============================
# Reset Button
# ===============================
if st.session_state.ready:
    if st.button("🔄 Reset Session"):
        st.session_state.index = None
        st.session_state.image_data = []
        st.session_state.ready = False
        st.experimental_rerun()
