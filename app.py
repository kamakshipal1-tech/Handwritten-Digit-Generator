import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# ---------------------------------------
# Page Configuration
# ---------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="✍️",
    layout="wide"
)

# ---------------------------------------
# Load Generator Model
# ---------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("generator.keras")

generator = load_model()

# ---------------------------------------
# Sidebar
# ---------------------------------------
with st.sidebar:
    st.header("📌 Project Details")

    st.markdown("""
**Model:** DCGAN

**Dataset:** MNIST

**Image Size:** 28 × 28

**Noise Dimension:** 100

**Framework:** TensorFlow / Keras

**Frontend:** Streamlit

**Output:** AI Generated Handwritten Digits
""")

# ---------------------------------------
# Title
# ---------------------------------------
st.title("✍️ AI Handwritten Digit Generator")
st.caption(
    "Generate realistic handwritten digits using a Deep Convolutional Generative Adversarial Network (DCGAN)."
)

# ---------------------------------------
# Metrics
# ---------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Dataset", "MNIST")
col2.metric("Training Images", "60,000")
col3.metric("Model", "DCGAN")

st.divider()

# ---------------------------------------
# Generate Images
# ---------------------------------------
if st.button("🎲 Generate New Digits", use_container_width=True):

    noise = tf.random.normal([16, 100])

    generated_images = generator(noise, training=False)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for i, ax in enumerate(axes.flatten()):
        image = generated_images[i, :, :, 0]
        image = image * 127.5 + 127.5

        ax.imshow(image, cmap="gray")
        ax.axis("off")

    plt.tight_layout()

    st.pyplot(fig)

    # -----------------------------
    # Download Button
    # -----------------------------
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)

    st.download_button(
        label="📥 Download Generated Digits",
        data=buf,
        file_name="generated_digits.png",
        mime="image/png"
    )

    plt.close(fig)

st.divider()

# ---------------------------------------
# About
# ---------------------------------------
st.subheader("📖 About this Project")

st.write("""
This application demonstrates a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **MNIST handwritten digit dataset**.

The model consists of two neural networks:

- **Generator:** Learns to transform random noise into realistic handwritten digit images.
- **Discriminator:** Learns to distinguish between real MNIST images and generated images.

Through adversarial training, both networks improve simultaneously, allowing the generator to create increasingly realistic handwritten digits.
""")

st.markdown("""
### 🛠 Technologies Used
- TensorFlow / Keras
- Streamlit
- NumPy
- Matplotlib
""")

st.divider()

# ---------------------------------------
# Footer
# ---------------------------------------
st.caption(
    "Developed by Kamakshi Pal | Deep Learning Project | TensorFlow • Streamlit"
)