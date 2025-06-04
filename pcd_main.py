import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from skimage import exposure
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
import io
import base64
from scipy import ndimage
import matplotlib.pyplot as plt
import requests
import json

st.set_page_config(
    page_title="Kelompok 9",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .process-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #764ba2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .process-title {
        font-weight: bold;
        color: #764ba2;
        font-size: 1.2em;
        margin-bottom: 0.5rem;
    }
    .process-description {
        color: #555;
        line-height: 1.6;
    }
    .formula {
        background: #e9ecef;
        padding: 0.5rem;
        border-radius: 5px;
        font-family: monospace;
        margin: 0.5rem 0;
        overflow-x: auto; /* Allow horizontal scrolling for long formulas */
    }
    .stButton > button {
        width: 100%;
        background-color: #667eea;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1.1em;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #764ba2;
    }
    .upload-section {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        border: 2px dashed #764ba2;
        margin-bottom: 1.5rem;
    }
    .result-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1.1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = "Theory"
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = {}


def normalize_image_0_1(image_array_uint8):
    """Normalizes pixel intensities from [0, 255] uint8 to [0, 1] float32."""
    return image_array_uint8.astype(np.float32) / 255.0

def adjust_brightness_linear_pcd(img_float_0_1, alpha=1.5, beta=0.1):
    """Adjusts image brightness linearly as in pcd_v2.py: alpha * img + beta."""
    return np.clip(alpha * img_float_0_1 + beta, 0, 1)

def adjust_brightness_log_pcd(img_float_0_1):
    """Adjusts image brightness logarithmically as in pcd_v2.py: log(1 + I) normalized."""
    # log1p(x) is log(1+x)
    # Normalize by dividing by log(1 + max_possible_value_in_0_1) which is log(1+1) = log(2)
    return np.log1p(img_float_0_1) / np.log1p(1.0)

def histogram_equalization_channel_pcd(channel_float_0_1):
    """Performs histogram equalization on a single channel (0-1 float)."""
    # Convert to 0-255 for histogram calculation, then back to 0-1
    channel_uint8 = np.clip(channel_float_0_1 * 255, 0, 255).astype(np.uint8)
    hist, bins = np.histogram(channel_uint8.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    
    # Handle cases where cdf_min might be 0 (e.g., uniform image)
    cdf_min = cdf.min()
    cdf_max = cdf.max()
    if cdf_max - cdf_min == 0:
        cdf_normalized = cdf * 0 # Avoid division by zero, result in flat image
    else:
        cdf_normalized = (cdf - cdf_min) * 255 / (cdf_max - cdf_min)
    
    eq_channel_uint8 = cdf_normalized[channel_uint8]
    return eq_channel_uint8.astype(np.float32) / 255.0

def enhance_contrast_pcd(img_float_0_1):
    """Enhances image contrast using Histogram Equalization per color channel."""
    channels = []
    for i in range(img_float_0_1.shape[2]): # Iterate over R, G, B channels
        ch_eq = histogram_equalization_channel_pcd(img_float_0_1[:, :, i])
        channels.append(ch_eq)
    return np.stack(channels, axis=2)

def reduce_noise_pcd(image_float_0_1, sigma=1.0):
    """Reduces noise using Gaussian Filtering (using OpenCV for efficiency)."""
    return cv2.GaussianBlur(image_float_0_1, (0, 0), sigmaX=sigma, sigmaY=sigma)

def grayscale_pcd(img_float_0_1):
    """Converts color image (0-1 float) to grayscale using luminance method."""
    # Formula: 0.2989*R + 0.5870*G + 0.1140*B
    return np.dot(img_float_0_1[..., :3], [0.2989, 0.5870, 0.1140])

def laplacian_filter_pcd():
    """Returns the Laplacian kernel as defined in pcd_v2.py."""
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]], dtype=np.float32)

def apply_filter_gray_pcd(img_gray_float_0_1, kernel):
    """Applies a filter to a grayscale image (0-1 float) using scipy.ndimage.convolve."""
    return ndimage.convolve(img_gray_float_0_1, kernel, mode='reflect')

def enhance_edges_pcd(img_float_0_1, factor=0.3):
    """Enhances edges using a Laplacian filter as in pcd_v2.py's sharpen_image_color."""
    # Convert to grayscale for Laplacian calculation
    gray = grayscale_pcd(img_float_0_1)
    
    # Apply Laplacian filter to grayscale image
    lap = apply_filter_gray_pcd(gray, laplacian_filter_pcd())
    
    # Sharpened image is original - factor * laplacian
    sharpened_img = img_float_0_1 - factor * np.stack([lap]*3, axis=2)
    
    return np.clip(sharpened_img, 0, 1)

def calculate_metrics(original_img_array_uint8, processed_img_array_uint8):
    """Calculates MSE, PSNR, and SSIM between two images (expects uint8, 0-255)."""
    
    # Ensure images are grayscale for SSIM if they are color, or convert to float
    if len(original_img_array_uint8.shape) == 3 and original_img_array_uint8.shape[2] == 3:
        original_gray = cv2.cvtColor(original_img_array_uint8, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original_img_array_uint8

    if len(processed_img_array_uint8.shape) == 3 and processed_img_array_uint8.shape[2] == 3:
        processed_gray = cv2.cvtColor(processed_img_array_uint8, cv2.COLOR_RGB2GRAY)
    else:
        processed_gray = processed_img_array_uint8

    # Ensure both images have the same dimensions for metric calculation
    if original_gray.shape != processed_gray.shape:
        processed_gray = cv2.resize(processed_gray, (original_gray.shape[1], original_gray.shape[0]), interpolation=cv2.INTER_AREA)

    mse = mean_squared_error(original_gray, processed_gray)
    psnr = peak_signal_noise_ratio(original_gray, processed_gray, data_range=255)
    
    # SSIM requires data_range to be specified
    ssim = structural_similarity(original_gray, processed_gray, data_range=255)
    
    return mse, psnr, ssim

# sidebar
with st.sidebar:
    st.markdown("## 📋 Navigation")
    
    page = st.radio(
        "Select Page:",
        ["📚 Theory & Processes", "🔧 Image Processing Tool"],
        index=0 if st.session_state.page == "Theory" else 1
    )
    
    # Update session state based on selection
    if page == "📚 Theory & Processes":
        st.session_state.page = "Theory"
    else:
        st.session_state.page = "Processing"
    
    st.markdown("---")
    
    if st.session_state.page == "Theory":
        st.markdown("""
        ### 📖 Proses yang dilakukan
        Berikut adalah beberapa tahapan pengaplikasian yang kami lakukan dalam pengolahan citra.
        """)
    else:
        st.markdown("""
        ### 🛠️ About This Page
        Upload your images and process them using the techniques described in the theory page.
        """)

# page 1
if st.session_state.page == "Theory":
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 Image Enhancement Theory & Processes</h1>
        <p>Kelompok 9 | Ulayya Azizna, Nisrina Irbah, Arief Kurniawan</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ## 🎯 Overview
    Banyak citra yang diambil pada kondisi pencahayaan rendah (low-light). Kami mengaplikasikan beberapa proses pengolahan citra untuk meningkatkan visibilitas citra, sehingga lebih mudah diaplikasikan pada tahap selanjutnya seperti deteksi objek. Dataset yang digunakan untuk pengujian adalah <a href="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset" target="_blank">Exclusively-Dark-Image-Dataset</a>.
    """, unsafe_allow_html=True)
    
    # Process 1: Pre-processing (Normalization)
    st.markdown("""
    <div class="process-card">
        <div class="process-title">1. Pre-processing: Normalisasi Intensitas Piksel</div>
        <div class="process-description">
            <p><strong>Tujuan:</strong> Mengubah nilai intensitas piksel dari rentang [0, 255] ke rentang standar [0, 1] (floating-point). Normalisasi ini penting agar semua operasi selanjutnya bekerja secara konsisten dalam skala intensitas piksel yang sama.</p>
            <p><strong>Metode:</strong> Normalisasi sederhana dengan pembagian.</p>
            <div class="formula">
                img_array = np.asarray(img).astype(np.float32) / 255.0
            </div>
            <p>Dimana img_array adalah nilai intensitas piksel asli.</p>
            <p><strong>Keuntungan:</strong> Memastikan konsistensi pemrosesan dan efektivitas algoritma yang sensitif terhadap skala input.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process 2: Brightness Adjustment
    st.markdown("""
    <div class="process-card">
        <div class="process-title">2. Brightness Adjustment: Linear/Logarithmic Transformation</div>
        <div class="process-description">
            <p>Penyesuaian kecerahan dilakukan untuk membantu memperjelas detail di area gelap atau <i>low-light</i>.</p>
            <p><strong>Linear Transformation:</strong></p>
            <div class="formula">
                def adjust_brightness_linear(img, alpha=1.5, beta=0.1):
                    return np.clip(alpha * img + beta, 0, 1)
            </div>
          <p><strong>Logarithmic Transformation:</strong></p>
            <div class="formula">
                def adjust_brightness_log(img):
                return np.clip(alpha * img + beta, 0, 1)
            </div>
            <p><strong>Aplikasi:</strong> Lebih cocok untuk menonjolkan detail halus di area gelap tanpa terlalu mencerahkan seluruh gambar.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process 3: Contrast Enhancement
    st.markdown("""
    <div class="process-card">
        <div class="process-title">3. Peningkatan Kontras: Histogram Equalization (Per Channel)</div>
        <div class="process-description">
            <p><strong>Prinsip:</strong> Dilakukan histogram equalization untuk setiap saluran warna (R, G, dan B) secara independen. Metode ini meningkatkan kontras gambar dengan meratakan distribusi histogram intensitas piksel pada masing-masing saluran RGB, membuat detail visual lebih tampak, terutama pada area gelap atau datar.</p>
            <div class="formula">
                def equalize_histogram(img):
                out = np.zeros_like(img)
                for channel in range(3):  # RGB
                    flat = img[:, :, channel].flatten()
                    hist, bins = np.histogram(flat, bins=256, range=[0,1])
                    cdf = hist.cumsum()
                    cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())
                    out[:, :, channel] = cdf[(img[:, :, channel] * 255).astype(np.uint8)]
                return out
            </div>
            <p><strong>Manfaat:</strong> Meningkatkan kontras global citra sambil mempertahankan keseimbangan warna dan memperluas rentang dinamis citra secara efektif.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process 4: Noise Reduction
    st.markdown("""
    <div class="process-card">
        <div class="process-title">4. Pengurangan Noise: Gaussian Filtering</div>
        <div class="process-description">
            <p><strong>Tujuan:</strong> Mengurangi noise (bintik-bintik kecil) dengan menerapkan blur halus menggunakan kernel Gaussian. Filter ini merata-ratakan nilai piksel tetangga berdasarkan distribusi Gaussian.</p>
            <p><strong>Fungsi Gaussian 2D:</strong></p>
            <div class="formula">
                def gaussian_kernel(size=5, sigma=1):
                    ax = np.linspace(-(size // 2), size // 2, size)
                    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
                    kernel = np.outer(gauss, gauss)
                    return kernel / np.sum(kernel)
            </div>
            <p><strong>Karakteristik:</strong> Filter linear yang efektif untuk penghalusan gambar dan mempertahankan tepi dengan parameter $\\sigma$ yang tepat.</p>
            <p><strong>Parameter sigma:</strong></p>
            <ul>
                <li>sigma kecil (0.5-1.0): Penghalusan minimal, detail terjaga.</li>
                <li>sigma sedang (1.0-2.0): Keseimbangan noise reduction dan detail.</li>
                <li>sigma besar (>2.0): Penghalusan maksimal, detail berkurang.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process 5: Edge Enhancement
    st.markdown("""
    <div class="process-card">
        <div class="process-title">5. Peningkatan Ketajaman: Laplacian Filter</div>
        <div class="process-description">
            <p><strong>Tujuan:</strong> Meningkatkan ketajaman tepi dengan menonjolkan transisi intensitas. Setelah blur (reduksi noise), langkah ini menegaskan kembali tepi agar gambar tidak terlihat lembek atau kusam.</p>
            <p><strong>Operator Laplacian:</strong> Mendeteksi perubahan mendadak intensitas piksel, yang menandakan tepi objek.</p>
            <p><strong>Kernel Laplacian (3×3) yang digunakan:</strong></p>
            <div class="formula">
                [[0  1  0]
                \n[1 -4  1]
                \n[0  1  0]]
            </div>
            <p><strong>Proses Sharpen:</strong></p>
            <div class="formula">
                def sharpen_image(img):
                    gray = grayscale(img)
                    lap = apply_filter_gray(gray, laplacian_filter())
                    sharp = gray - 0.3 * lap  
                    sharp = np.clip(sharp, 0, 1)
                    return np.stack([sharp]*3, axis=2)  
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Process 6: Post-processing (Evaluation)
    st.markdown("""
    <div class="process-card">
        <div class="process-title">6. Post-processing: Evaluasi Kualitas Citra</div>
        <div class="process-description">
            <p><strong>Metrik Evaluasi:</strong> Digunakan untuk mengukur perbedaan antara citra asli dan citra yang telah diproses.</p>
            <p><strong>1. Mean Squared Error (MSE):</strong> Mengukur rata-rata kuadrat perbedaan antara nilai piksel.</p>
            <div class="formula">
                MSE mean_squared_error(original_gray, processed_gray)
            </div>    
            <p><strong>2. Peak Signal-to-Noise Ratio (PSNR):</strong> Mengukur rasio antara daya maksimum sinyal dan daya noise yang mempengaruhi fidelitasnya.</p>
            <div class="formula">
                psnr = peak_signal_noise_ratio(original_gray, processed_gray, data_range=255)
            </div>
            <p><strong>3. Structural Similarity Index (SSIM):</strong> Mengukur kesamaan struktural antara dua citra, mempertimbangkan luminansi, kontras, dan struktur.</p>
            <div class="formula">
                ssim = structural_similarity(original_gray, processed_gray, data_range=255)
            </div>
            <p><strong>Interpretasi:</strong></p>
            <ul>
                <li><strong>MSE:</strong> Semakin rendah nilainya, semakin baik (0 = identik).</li>
                <li><strong>PSNR:</strong> Semakin tinggi nilainya, semakin baik (biasanya >30 dB dianggap kualitas baik).</li>
                <li><strong>SSIM:</strong> Rentang [0,1], semakin tinggi nilainya (mendekati 1), semakin tinggi kesamaan struktural.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
   # Processing Pipeline Diagram Placeholder
    st.markdown("## 🔄 Processing Pipeline")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #cfd8dc; border-radius: 8px; border: 1px solid #90a4ae; color: #333333;">
            <h3>Alur Pemrosesan Gambar</h3>
            <p>
                <code>Citra Asli</code> ➡️ <code>Pre-processing (Normalisasi 0-1)</code> ➡️ <code>Penyesuaian Kecerahan (Linear/Logaritmik)</code> ➡️ <code>Peningkatan Kontras (Histogram Equalization Per Channel)</code> ➡️ <code>Pengurangan Noise (Gaussian)</code> ➡️ <code>Peningkatan Ketajaman (Laplacian)</code> ➡️ <code>Citra yang Diproses</code>
            </p>
        </div>
        """, unsafe_allow_html=True)

# page 2
elif st.session_state.page == "Processing":
    st.markdown("""
    <div class="main-header">
        <h1>🔧 Image Processing Tool</h1>
        <p>Upload your image and apply various enhancement techniques</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⬆️ Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Read the image as uint8 (0-255)
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Convert BGR to RGB for consistent display with Streamlit/PIL
        original_image_rgb_uint8 = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)
        
        st.session_state.uploaded_image = original_image_rgb_uint8 # Store original as uint8 for metrics
        
        # Normalize to 0-1 float for processing functions
        st.session_state.image_for_processing = normalize_image_0_1(original_image_rgb_uint8) 
        st.session_state.processed_image_float = st.session_state.image_for_processing.copy() # Initialize processed with original (0-1 float)

        st.success("Image uploaded successfully!")

        st.markdown("---")
        st.markdown("## ⚙️ Enhancement Parameters")

        # Create tabs for different enhancement types
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Pre-processing", "Brightness", "Contrast", "Noise Reduction", "Edge Enhancement"
        ])
        
        with tab1:
            st.markdown("### Normalisasi Intensitas Piksel")
            st.info("This step normalizes pixel values to the standard [0, 1] float range for internal processing.")

        with tab2:
            st.markdown("### Brightness Adjustment")
            brightness_method = st.radio(
                "Select Brightness Method:",
                ("Linear Transformation", "Logarithmic Transformation"),
                key="brightness_method"
            )
            if brightness_method == "Linear Transformation":
                brightness_alpha = st.slider(
                    "Linear Alpha (Contrast Factor):",
                    min_value=0.1, max_value=3.0, value=1.5, step=0.1, key="brightness_alpha"
                )
                brightness_beta = st.slider(
                    "Linear Beta (Brightness Offset):",
                    min_value=-0.5, max_value=0.5, value=0.1, step=0.05, key="brightness_beta"
                )
                st.info("Formula: $s = \\alpha \\times r + \\beta$.")
            else: # Logarithmic Transformation
                st.info("Formula: $s = \\frac{\\log(1 + r)}{\\log(1 + r_{max})}$. This enhances dark areas.")

        with tab3:
            st.markdown("### Contrast Enhancement (Per Channel Histogram Equalization)")
            st.info("Applies Histogram Equalization independently to each R, G, and B channel to enhance global contrast.")
            apply_contrast = st.checkbox("Apply Per-Channel Histogram Equalization", value=True, key="apply_contrast")

        with tab4:
            st.markdown("### Noise Reduction (Gaussian Filter)")
            noise_sigma = st.slider(
                "Gaussian Blur Sigma ($\\sigma$):",
                min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="noise_sigma"
            )
            st.info("Higher $\\sigma$ means more blurring and noise reduction. Set to 0 for no effect.")

        with tab5:
            st.markdown("### Edge Enhancement (Laplacian Filter)")
            edge_factor = st.slider(
                "Edge Enhancement Factor ($c$):",
                min_value=0.0, max_value=2.0, value=0.3, step=0.05, key="edge_factor"
            )
            st.info("Higher factor means stronger edge sharpening. Set to 0 for no effect. Formula: $g(x,y) = f(x,y) - c \\times \\nabla^2f_{gray}(x,y)$.")

        st.markdown("---")
        if st.button("Process Image", key="process_button"):
            if st.session_state.image_for_processing is not None:
                current_image_float = st.session_state.image_for_processing.copy()
                
                with st.spinner("Processing image..."):
                    # Apply enhancements in sequence
                    
                    # Brightness
                    if brightness_method == "Linear Transformation":
                        current_image_float = adjust_brightness_linear_pcd(
                            current_image_float, 
                            alpha=brightness_alpha, 
                            beta=brightness_beta
                        )
                    else: # Logarithmic Transformation
                        current_image_float = adjust_brightness_log_pcd(current_image_float)
                    
                    # Contrast
                    if apply_contrast:
                        current_image_float = enhance_contrast_pcd(current_image_float)
                        
                    # Noise Reduction
                    if noise_sigma > 0:
                        current_image_float = reduce_noise_pcd(current_image_float, sigma=noise_sigma)
                        
                    # Edge Enhancement
                    if edge_factor > 0:
                        current_image_float = enhance_edges_pcd(current_image_float, factor=edge_factor)
                    
                    st.session_state.processed_image_float = current_image_float

                    # Convert processed image back to uint8 for display and metric calculation
                    processed_image_uint8 = np.clip(st.session_state.processed_image_float * 255, 0, 255).astype(np.uint8)

                    # Calculate metrics using the original uint8 image and the final processed uint8 image
                    mse, psnr, ssim = calculate_metrics(
                        st.session_state.uploaded_image, # Original uint8
                        processed_image_uint8             # Processed uint8
                    )
                    st.session_state.processing_results = {
                        "MSE": f"{mse:.4f}",
                        "PSNR": f"{psnr:.2f} dB",
                        "SSIM": f"{ssim:.4f}"
                    }
                st.success("Image processing complete!")
            else:
                st.warning("Please upload an image first!")

        st.markdown("---")
        st.markdown("## ✨ Results")

        if st.session_state.processed_image_float is not None:
            # Convert processed image back to uint8 for display
            display_processed_image_uint8 = np.clip(st.session_state.processed_image_float * 255, 0, 255).astype(np.uint8)

            col_orig, col_proc = st.columns(2)
            with col_orig:
                st.subheader("Original Image")
                st.image(st.session_state.uploaded_image, caption="Original Image", use_container_width=True)
            with col_proc:
                st.subheader("Processed Image")
                st.image(display_processed_image_uint8, caption="Processed Image", use_container_width=True)
            
            st.markdown("### Quality Metrics (Processed vs. Original)")
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Mean Squared Error (MSE)", st.session_state.processing_results.get("MSE", "N/A"))
            with metrics_col2:
                st.metric("Peak Signal-to-Noise Ratio (PSNR)", st.session_state.processing_results.get("PSNR", "N/A"))
            with metrics_col3:
                st.metric("Structural Similarity Index (SSIM)", st.session_state.processing_results.get("SSIM", "N/A"))
            
            st.markdown("""
            <div class="process-description">
                <p><strong>Interpretation of Metrics:</strong></p>
                <ul>
                    <li><strong>MSE:</strong> Lower is better (0 means identical images).</li>
                    <li><strong>PSNR:</strong> Higher is better (typically >30 dB is considered good quality).</li>
                    <li><strong>SSIM:</strong> Closer to 1 indicates higher structural similarity.</li>
                </ul>
                <p>These metrics quantify the difference between the original and processed image. A "better" enhancement might not always mean a higher PSNR/SSIM, as some enhancements intentionally alter the image for visual clarity rather than strict similarity.</p>
            </div>
            """, unsafe_allow_html=True)

            # Option to download processed image
            processed_pil_image = Image.fromarray(display_processed_image_uint8)
            buf = io.BytesIO()
            processed_pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="enhanced_image.png",
                mime="image/png",
                key="download_button"
            )
        else:
            st.info("Upload an image to see the results.")