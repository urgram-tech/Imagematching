
import streamlit as st
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="AI Image Matcher", layout="wide")

st.title("🧠 AI Image Matching App")

# ============================
# LOAD MODEL (only once)
# ============================
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_model()

# ============================
# UPLOAD DATABASE IMAGES
# ============================
st.header("1️⃣ Upload Database Images")
database_files = st.file_uploader(
    "Upload multiple images",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if database_files:

    st.success(f"{len(database_files)} images uploaded")

    image_names = []
    embeddings = []

    for file in database_files:
        img = Image.open(file).convert("RGB")
        emb = model.encode(img)
        embeddings.append(emb)
        image_names.append(file.name)

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # ============================
    # UPLOAD QUERY IMAGE
    # ============================
    st.header("2️⃣ Upload Image to Compare")
    query_file = st.file_uploader(
        "Upload query image",
        type=["png", "jpg", "jpeg"]
    )

    if query_file:
        query_img = Image.open(query_file).convert("RGB")

        st.image(query_img, caption="Query Image", width=300)

        query_embedding = model.encode(query_img)
        query_embedding = np.array([query_embedding])

        D, I = index.search(query_embedding, 3)

        st.header("🔍 Top Matches")

        for rank, idx in enumerate(I[0]):
            distance = D[0][rank]
            similarity = 1 / (1 + distance)
            percentage = round(similarity * 100, 2)

            st.subheader(f"#{rank+1} - {image_names[idx]}")
            st.write(f"Match: {percentage}%")
            st.image(database_files[idx], width=300)
