from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
from gtts import gTTS
from pydub import AudioSegment
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly, stft, istft

from PIL import Image, ImageEnhance, ImageDraw
import cv2
import fitz  # PyMuPDF

st.set_page_config(
    page_title="How Might They Hear It?",
    page_icon="🎧",
    layout="wide",
)

FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
DEFAULT_LEFT = [10, 15, 25, 45, 65, 70]
DEFAULT_RIGHT = [10, 15, 25, 50, 70, 75]
TARGET_SR = 44100

# -----------------------------
# Audio helpers
# -----------------------------
def to_mono_float(x: np.ndarray, sr: int):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if x.size == 0:
        raise ValueError("The audio file appears to be empty.")
    peak = float(np.max(np.abs(x)) + 1e-9)
    if peak > 1.0:
        x = x / peak
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g)
        sr = TARGET_SR
    return x.astype(np.float32), sr

def decode_audio_bytes(audio_bytes: bytes, suffix: str = ".wav"):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as src:
        src.write(audio_bytes)
        src_path = Path(src.name)
    wav_path = src_path.with_suffix(".decoded.wav")
    try:
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_channels(1)
        audio.export(wav_path, format="wav")
        x, sr = sf.read(wav_path, always_2d=False)
        return to_mono_float(x, sr)
    finally:
        src_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

def wav_bytes(x: np.ndarray, sr: int):
    bio = io.BytesIO()
    sf.write(bio, x, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()

def stereo_wav_bytes(left: np.ndarray, right: np.ndarray, sr: int):
    n = min(len(left), len(right))
    stereo = np.column_stack([left[:n], right[:n]])
    bio = io.BytesIO()
    sf.write(bio, stereo, sr, format="WAV", subtype="PCM_16")
    return bio.getvalue()

# -----------------------------
# Hearing simulation
# -----------------------------
def interp_hl(freq_bins, thresholds):
    f = np.maximum(freq_bins, FREQS[0])
    return np.interp(
        np.log2(f),
        np.log2(FREQS),
        np.asarray(thresholds, float),
        left=float(thresholds[0]),
        right=float(thresholds[-1]),
    )

def simulate_ear(x: np.ndarray, sr: int, thresholds, speech_level_db=65.0):
    x = np.asarray(x, np.float32)
    original_len = len(x)
    if len(x) < 4096:
        x = np.pad(x, (0, 4096 - len(x)))

    f, _, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536, boundary="zeros")
    mag = np.abs(Z) + 1e-10
    phase = np.angle(Z)
    hl = interp_hl(f, thresholds)[:, None]

    rms = np.sqrt(np.mean(x * x) + 1e-12)
    ref_dbfs = 20 * np.log10(rms + 1e-12)
    bin_dbfs = 20 * np.log10(mag + 1e-12)
    est_spl = speech_level_db + (bin_dbfs - ref_dbfs)

    sensation = est_spl - hl
    audibility_gain_db = np.where(
        sensation <= -10,
        -80,
        np.where(
            sensation < 0,
            -35 + 3.5 * sensation,
            np.where(
                sensation < 25,
                -0.55 * hl * (1 - sensation / 25),
                0.0,
            ),
        ),
    )
    audibility_gain_db = np.clip(audibility_gain_db, -80, 0)
    mag2 = mag * (10.0 ** (audibility_gain_db / 20.0))

    mean_hl = float(np.mean(thresholds))
    sigma_bins = np.clip((mean_hl - 20.0) / 18.0, 0.0, 4.0)
    if sigma_bins > 0.05:
        logmag = np.log(mag2 + 1e-12)
        blurred = gaussian_filter1d(logmag, sigma=sigma_bins, axis=0, mode="nearest")
        mix = np.clip((mean_hl - 20) / 60, 0, 0.75)
        mag2 = np.exp((1 - mix) * logmag + mix * blurred)

    exponent = np.clip(1.0 - max(mean_hl - 20.0, 0.0) / 180.0, 0.65, 1.0)
    scale = np.percentile(mag2, 95) + 1e-10
    mag2 = scale * np.power(np.maximum(mag2 / scale, 0), exponent)

    Z2 = mag2 * np.exp(1j * phase)
    _, y = istft(Z2, fs=sr, nperseg=2048, noverlap=1536, input_onesided=True)
    y = y[:original_len]

    p = np.max(np.abs(y)) + 1e-9
    if p > 0.98:
        y = y * (0.98 / p)
    return y.astype(np.float32)

# -----------------------------
# TTS
# -----------------------------
def generate_tts(text: str):
    text = (text or "").strip()
    if not text:
        raise ValueError("Type a sentence first.")
    mp3 = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(mp3)
    mp3.seek(0)
    return decode_audio_bytes(mp3.getvalue(), ".mp3")

# -----------------------------
# Audiogram helpers
# -----------------------------
def load_audiogram_image_from_bytes(data: bytes, filename: str = "image.jpg"):
    name = filename.lower()
    if name.endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    return Image.open(io.BytesIO(data)).convert("RGB")

def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    rect[1] = pts[np.argmin(d)]   # top-right
    rect[3] = pts[np.argmax(d)]   # bottom-left
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_w = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_h = int(max(height_a, height_b))

    if max_w < 50 or max_h < 50:
        return image

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (max_w, max_h))

def auto_straighten_page(img: Image.Image):
    """
    Attempt to find the photographed sheet/page and correct perspective.
    Falls back safely to the original image.
    """
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    scale = min(1400 / max(h, w), 1.0)
    small = cv2.resize(arr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else arr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

    image_area = small.shape[0] * small.shape[1]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(approx) > image_area * 0.18:
            pts = approx.reshape(4, 2).astype(np.float32)
            warped = four_point_transform(small, pts)
            if warped.size > 0:
                return Image.fromarray(warped)

    return img

def enhance_for_detection(img: Image.Image):
    img = img.convert("RGB")
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.15)
    return img

def detect_plot_region(img: Image.Image):
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, threshold=100,
        minLineLength=max(80, min(arr.shape[:2])//5),
        maxLineGap=12
    )

    h, w = gray.shape
    verticals, horizontals = [], []

    if lines is not None:
        for l in lines[:,0]:
            x1,y1,x2,y2 = map(int,l)
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dy > 4*max(dx,1):
                verticals.append((x1,y1,x2,y2))
            elif dx > 4*max(dy,1):
                horizontals.append((x1,y1,x2,y2))

    if verticals and horizontals:
        xs = [int((x1+x2)/2) for x1,y1,x2,y2 in verticals]
        ys = [int((y1+y2)/2) for x1,y1,x2,y2 in horizontals]

        x_candidates = [x for x in xs if 0.08*w < x < 0.92*w]
        y_candidates = [y for y in ys if 0.08*h < y < 0.92*h]

        if len(x_candidates) >= 2 and len(y_candidates) >= 2:
            left = int(np.percentile(x_candidates, 10))
            right = int(np.percentile(x_candidates, 90))
            top = int(np.percentile(y_candidates, 10))
            bottom = int(np.percentile(y_candidates, 90))

            if right-left > 0.35*w and bottom-top > 0.30*h:
                return (left, top, right, bottom)

    return (int(0.12*w), int(0.18*h), int(0.90*w), int(0.86*h))

def color_masks(crop_rgb):
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 65, 55]), np.array([14, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([165, 65, 55]), np.array([180, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    blue = cv2.inRange(hsv, np.array([88, 50, 35]), np.array([145, 255, 255]))

    kernel = np.ones((3,3), np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    return red, blue

def detect_points(mask, crop_w, crop_h):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    pts = []
    min_area = max(4, int(crop_w*crop_h*0.00001))
    max_area = max(1200, int(crop_w*crop_h*0.02))

    for i in range(1,n):
        x,y,w,h,area = stats[i]
        if min_area <= area <= max_area:
            cx, cy = centroids[i]
            if 0.02*crop_w < cx < 0.98*crop_w and 0.02*crop_h < cy < 0.98*crop_h:
                pts.append((float(cx), float(cy), float(area)))
    return pts

def extract_thresholds_from_image(img: Image.Image):
    straight = auto_straighten_page(img)
    processed = enhance_for_detection(straight)

    box = detect_plot_region(processed)
    arr = np.array(processed.convert("RGB"))
    x1,y1,x2,y2 = box
    crop = arr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]

    red_mask, blue_mask = color_masks(crop)
    red_pts = detect_points(red_mask, cw, ch)
    blue_pts = detect_points(blue_mask, cw, ch)

    min_f, max_f = 125.0, 8000.0
    target_x = {
        int(f): (np.log2(f/min_f) / np.log2(max_f/min_f)) * cw
        for f in FREQS
    }

    def map_points(points):
        out = {}
        for f in FREQS.astype(int):
            tx = target_x[f]
            max_dx = cw * 0.065
            candidates = [p for p in points if abs(p[0]-tx) <= max_dx]
            if not candidates:
                out[f] = None
                continue

            candidates.sort(key=lambda p: (-p[2], abs(p[0]-tx)))
            cx, cy, area = candidates[0]

            db = -10 + (cy / max(ch-1,1)) * 130
            db = int(round(db/5)*5)
            out[f] = int(np.clip(db, -10, 120))
        return out

    right = map_points(red_pts)
    left = map_points(blue_pts)

    preview = processed.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle(box, outline="green", width=4)

    def draw_pts(points, color):
        for cx,cy,_ in points:
            gx, gy = x1+cx, y1+cy
            r = 7
            draw.ellipse((gx-r, gy-r, gx+r, gy+r), outline=color, width=3)

    draw_pts(red_pts, "red")
    draw_pts(blue_pts, "blue")

    return left, right, straight, preview, len(blue_pts), len(red_pts)

def apply_detected_thresholds(img):
    left_detected, right_detected, straight, preview, nblue, nred = extract_thresholds_from_image(img)

    found = 0
    for i, f in enumerate(FREQS.astype(int)):
        if left_detected[f] is not None:
            st.session_state.left[i] = left_detected[f]
            found += 1
        if right_detected[f] is not None:
            st.session_state.right[i] = right_detected[f]
            found += 1

    st.session_state.extraction_straight = straight
    st.session_state.extraction_preview = preview
    st.session_state.extraction_message = (
        f"Detected {found} threshold values. "
        f"Found {nblue} blue and {nred} red candidate marks."
    )
    st.session_state.simulated_bytes = None

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem;}
      h1 {letter-spacing: -0.03em;}
      .callout {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 8px 0 18px 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("How Might They Hear It?")
st.caption("Audiogram-driven speech simulation — mobile-friendly prototype")

st.markdown(
    """
    <div class="callout">
    <strong>Educational prototype, not a diagnostic or clinical tool.</strong>
    An audiogram cannot fully describe a person's subjective hearing.
    Camera/PDF extraction is approximate and must be checked before use.
    </div>
    """,
    unsafe_allow_html=True,
)

for key, value in {
    "left": list(DEFAULT_LEFT),
    "right": list(DEFAULT_RIGHT),
    "source_wav": None,
    "source_sr": TARGET_SR,
    "source_label": None,
    "simulated_bytes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Audiogram
# -----------------------------
st.subheader("1. Add the audiogram")

camera_tab, upload_tab, manual_tab = st.tabs(
    ["Take a photo", "Upload PDF/photo", "Enter manually"]
)

with camera_tab:
    st.write(
        "On a phone, tap below and photograph the audiogram. "
        "Try to fill the frame with the page and avoid glare."
    )

    camera_photo = st.camera_input("Take a picture of the audiogram")

    if camera_photo is not None:
        try:
            img = load_audiogram_image_from_bytes(
                camera_photo.getvalue(),
                "camera.jpg",
            )
            st.image(img, caption="Camera photo", use_container_width=True)

            if st.button("Read camera photo", type="primary", key="read_camera"):
                apply_detected_thresholds(img)

        except Exception as e:
            st.error(f"I couldn't read that camera photo: {e}")

with upload_tab:
    ag_file = st.file_uploader(
        "Upload an audiogram",
        type=["pdf", "png", "jpg", "jpeg", "webp"],
        key="audiogram_file",
    )

    if ag_file is not None:
        try:
            img = load_audiogram_image_from_bytes(
                ag_file.getvalue(),
                ag_file.name,
            )
            st.image(img, caption="Uploaded audiogram", use_container_width=True)

            if st.button("Read uploaded audiogram", type="primary", key="read_upload"):
                apply_detected_thresholds(img)

        except Exception as e:
            st.error(f"I couldn't read that audiogram: {e}")

with manual_tab:
    st.write("Enter the threshold values directly below.")

if "extraction_preview" in st.session_state:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Straightened image**")
        st.image(
            st.session_state.extraction_straight,
            use_container_width=True,
        )

    with c2:
        st.markdown("**Detected chart and marks**")
        st.image(
            st.session_state.extraction_preview,
            use_container_width=True,
        )

    st.info(st.session_state.extraction_message)
    st.warning(
        "Please check every extracted threshold below before simulation. "
        "Camera angle, glare and unusual audiogram formats can cause errors."
    )

st.markdown("#### Confirm thresholds (dB HL)")
left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Left ear**")
    left_vals = []
    for i, f in enumerate(FREQS.astype(int)):
        left_vals.append(
            st.number_input(
                f"{f if f < 1000 else str(int(f/1000)) + ' k'}Hz",
                min_value=-10,
                max_value=120,
                value=int(st.session_state.left[i]),
                step=5,
                key=f"left_confirm_{f}",
            )
        )

with right_col:
    st.markdown("**Right ear**")
    right_vals = []
    for i, f in enumerate(FREQS.astype(int)):
        right_vals.append(
            st.number_input(
                f"{f if f < 1000 else str(int(f/1000)) + ' k'}Hz",
                min_value=-10,
                max_value=120,
                value=int(st.session_state.right[i]),
                step=5,
                key=f"right_confirm_{f}",
            )
        )

st.caption(
    "Common convention: red = right ear, blue = left ear. "
    "Always verify against the original audiogram."
)

# -----------------------------
# Speech input
# -----------------------------
st.divider()
st.subheader("2. Choose what you want them to hear")

source_tab1, source_tab2, source_tab3 = st.tabs(
    ["Type a sentence", "Record your voice", "Upload audio"]
)

with source_tab1:
    text = st.text_area(
        "Sentence",
        value="Can you please put your shoes on?",
        height=90,
    )

    if st.button("Generate normal speech", type="primary"):
        try:
            x, sr = generate_tts(text)
            st.session_state.source_wav = x
            st.session_state.source_sr = sr
            st.session_state.source_label = "Generated speech"
            st.session_state.simulated_bytes = None
            st.success("Speech generated.")
        except Exception as e:
            st.error("I couldn't generate speech right now. Try recording or uploading audio.")
            st.caption(str(e))

    st.caption(
        "Typed text is sent to Google's text-to-speech service. "
        "Do not include identifying patient information."
    )

with source_tab2:
    recording = st.audio_input("Record a sentence", sample_rate=16000)

    if recording is not None:
        try:
            x, sr = decode_audio_bytes(recording.getvalue(), ".wav")
            st.session_state.source_wav = x
            st.session_state.source_sr = sr
            st.session_state.source_label = "Your recording"
            st.session_state.simulated_bytes = None
            st.success("Recording ready.")
        except Exception as e:
            st.error(f"Couldn't read that recording: {e}")

with source_tab3:
    uploaded = st.file_uploader(
        "Upload WAV, MP3, M4A or similar audio",
        type=["wav", "mp3", "m4a", "aac", "ogg", "flac"],
        key="speech_upload",
    )

    if uploaded is not None:
        try:
            suffix = "." + uploaded.name.split(".")[-1].lower()
            x, sr = decode_audio_bytes(uploaded.getvalue(), suffix)
            st.session_state.source_wav = x
            st.session_state.source_sr = sr
            st.session_state.source_label = uploaded.name
            st.session_state.simulated_bytes = None
            st.success("Audio ready.")
        except Exception as e:
            st.error(f"Couldn't read that audio file: {e}")

if st.session_state.source_wav is not None:
    st.markdown("**Normal speech**")
    st.audio(
        wav_bytes(st.session_state.source_wav, st.session_state.source_sr),
        format="audio/wav",
    )

# -----------------------------
# Simulation
# -----------------------------
st.divider()
st.subheader("3. Hear the approximation")

speech_level = st.slider(
    "Assumed speech level at the listener (dB SPL)",
    min_value=45,
    max_value=85,
    value=65,
    step=1,
)

if st.button("Simulate hearing", type="primary", use_container_width=True):
    if st.session_state.source_wav is None:
        st.warning("Generate, record, or upload some speech first.")
    else:
        with st.spinner("Creating the simulation…"):
            x = st.session_state.source_wav
            sr = st.session_state.source_sr
            yl = simulate_ear(x, sr, left_vals, speech_level)
            yr = simulate_ear(x, sr, right_vals, speech_level)
            st.session_state.simulated_bytes = stereo_wav_bytes(yl, yr, sr)

if st.session_state.simulated_bytes is not None:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Normal")
        st.audio(
            wav_bytes(st.session_state.source_wav, st.session_state.source_sr),
            format="audio/wav",
        )

    with c2:
        st.markdown("#### Simulated")
        st.audio(
            st.session_state.simulated_bytes,
            format="audio/wav",
        )
        st.caption("Stereo: left-ear simulation / right-ear simulation.")

    st.info("For left/right differences, listen through headphones.")

    st.download_button(
        "Download simulated WAV",
        data=st.session_state.simulated_bytes,
        file_name="simulated_hearing.wav",
        mime="audio/wav",
    )

with st.expander("Camera and scientific limitations"):
    st.markdown(
        """
        **Camera reader:** the app attempts to find the photographed page,
        correct perspective, improve contrast, locate the audiogram grid,
        and identify conventional red/right and blue/left marks.

        It will not work reliably for every audiogram. Glare, shadows,
        monochrome charts, handwriting, unusual symbols, masking/bone-conduction
        notation and non-standard graph layouts can all cause errors.

        **Hearing simulation:** this remains an educational approximation,
        not a validated reconstruction of an individual's subjective hearing.
        """
    )

st.divider()
st.caption(
    "Prototype V3 · Do not upload names, dates of birth, patient numbers, "
    "or other identifying health information."
)
