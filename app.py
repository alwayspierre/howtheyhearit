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

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="How Might They Hear It?",
    page_icon="🎧",
    layout="wide",
)

FREQS = np.array([250, 500, 1000, 2000, 4000, 8000], dtype=float)
DEFAULT_LEFT = [10, 15, 25, 45, 65, 70]
DEFAULT_RIGHT = [10, 15, 25, 50, 70, 75]
TARGET_SR = 44100

PROFILES = {
    "Sloping moderate–severe loss": (DEFAULT_LEFT, DEFAULT_RIGHT),
    "Mild high-frequency loss": (
        [10, 10, 15, 20, 35, 45],
        [10, 10, 15, 20, 35, 45],
    ),
    "Flat moderate loss": (
        [45, 45, 45, 45, 45, 45],
        [45, 45, 45, 45, 45, 45],
    ),
    "Normal / near-normal": (
        [5, 5, 10, 10, 10, 10],
        [5, 5, 10, 10, 10, 10],
    ),
}


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
    """Decode common browser/upload audio formats through pydub/ffmpeg."""
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
    """
    Educational hearing-loss approximation.

    Approximates:
    1) frequency-dependent audibility from the audiogram,
    2) spectral smearing as loss increases,
    3) altered effective dynamic range / recruitment.

    This is NOT a validated clinical perceptual model.
    """
    x = np.asarray(x, np.float32)
    original_len = len(x)
    if len(x) < 4096:
        x = np.pad(x, (0, 4096 - len(x)))

    f, _, Z = stft(
        x,
        fs=sr,
        nperseg=2048,
        noverlap=1536,
        boundary="zeros",
    )

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
        blurred = gaussian_filter1d(
            logmag,
            sigma=sigma_bins,
            axis=0,
            mode="nearest",
        )
        mix = np.clip((mean_hl - 20) / 60, 0, 0.75)
        mag2 = np.exp((1 - mix) * logmag + mix * blurred)

    exponent = np.clip(
        1.0 - max(mean_hl - 20.0, 0.0) / 180.0,
        0.65,
        1.0,
    )
    scale = np.percentile(mag2, 95) + 1e-10
    mag2 = scale * np.power(np.maximum(mag2 / scale, 0), exponent)

    Z2 = mag2 * np.exp(1j * phase)
    _, y = istft(
        Z2,
        fs=sr,
        nperseg=2048,
        noverlap=1536,
        input_onesided=True,
    )

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

    # gTTS sends the typed text to Google's speech service.
    mp3 = io.BytesIO()
    gTTS(text=text, lang="en", slow=False).write_to_fp(mp3)
    mp3.seek(0)
    return decode_audio_bytes(mp3.getvalue(), ".mp3")


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
      .block-container {max-width: 1120px; padding-top: 2.2rem;}
      h1 {letter-spacing: -0.03em;}
      .subtle {color: #666; font-size: 0.92rem;}
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
st.caption("Audiogram-driven speech simulation — web prototype")

st.markdown(
    """
    <div class="callout">
    <strong>Educational prototype, not a diagnostic or clinical tool.</strong>
    An audiogram cannot fully describe a person's subjective hearing.
    This simulation is an approximation intended for empathy, education,
    and research validation.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Session state
# -----------------------------
if "profile_name" not in st.session_state:
    st.session_state.profile_name = "Sloping moderate–severe loss"
if "left" not in st.session_state:
    st.session_state.left = list(DEFAULT_LEFT)
if "right" not in st.session_state:
    st.session_state.right = list(DEFAULT_RIGHT)
if "source_wav" not in st.session_state:
    st.session_state.source_wav = None
if "source_sr" not in st.session_state:
    st.session_state.source_sr = TARGET_SR
if "source_label" not in st.session_state:
    st.session_state.source_label = None
if "simulated_bytes" not in st.session_state:
    st.session_state.simulated_bytes = None


# -----------------------------
# Source speech
# -----------------------------
st.subheader("1. Choose what you want them to hear")

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
            st.error(
                "I couldn't generate speech right now. "
                "You can still use Record your voice or Upload audio."
            )
            st.caption(str(e))

    st.caption(
        "Privacy note: typed text is sent to Google's text-to-speech service "
        "to generate the voice. Do not include identifying patient information."
    )

with source_tab2:
    recording = st.audio_input(
        "Record a sentence",
        sample_rate=16000,
    )
    if recording is not None:
        try:
            recording_bytes = recording.getvalue()
            x, sr = decode_audio_bytes(recording_bytes, ".wav")
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
    normal_bytes = wav_bytes(
        st.session_state.source_wav,
        st.session_state.source_sr,
    )
    st.audio(normal_bytes, format="audio/wav")
    st.caption(st.session_state.source_label or "Source audio")


# -----------------------------
# Audiogram
# -----------------------------
st.divider()
st.subheader("2. Enter the audiogram")

profile_name = st.selectbox(
    "Start with an example profile",
    list(PROFILES.keys()),
    index=list(PROFILES.keys()).index(st.session_state.profile_name),
)

if profile_name != st.session_state.profile_name:
    st.session_state.profile_name = profile_name
    l, r = PROFILES[profile_name]
    st.session_state.left = list(l)
    st.session_state.right = list(r)
    st.session_state.simulated_bytes = None
    st.rerun()

st.caption(
    "Replace these example values with the hearing thresholds from the audiogram, in dB HL."
)

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("#### Left ear")
    left_vals = []
    for i, f in enumerate(FREQS.astype(int)):
        left_vals.append(
            st.number_input(
                f"{f if f < 1000 else str(int(f/1000)) + ' k'}Hz",
                min_value=-10,
                max_value=120,
                value=int(st.session_state.left[i]),
                step=5,
                key=f"left_{f}",
            )
        )

with right_col:
    st.markdown("#### Right ear")
    right_vals = []
    for i, f in enumerate(FREQS.astype(int)):
        right_vals.append(
            st.number_input(
                f"{f if f < 1000 else str(int(f/1000)) + ' k'}Hz",
                min_value=-10,
                max_value=120,
                value=int(st.session_state.right[i]),
                step=5,
                key=f"right_{f}",
            )
        )

speech_level = st.slider(
    "Assumed speech level at the listener (dB SPL)",
    min_value=45,
    max_value=85,
    value=65,
    step=1,
    help="Around 60–65 dB SPL is a typical conversational level at close range.",
)


# -----------------------------
# Simulation
# -----------------------------
st.divider()
st.subheader("3. Hear the approximation")

if st.button("Simulate hearing", type="primary", use_container_width=True):
    if st.session_state.source_wav is None:
        st.warning("Generate, record, or upload some speech first.")
    else:
        with st.spinner("Creating the simulation…"):
            x = st.session_state.source_wav
            sr = st.session_state.source_sr
            yl = simulate_ear(x, sr, left_vals, speech_level)
            yr = simulate_ear(x, sr, right_vals, speech_level)
            st.session_state.simulated_bytes = stereo_wav_bytes(
                yl,
                yr,
                sr,
            )

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

with st.expander("What is this prototype actually doing?"):
    st.markdown(
        """
        This V1 uses the audiogram to approximate **frequency-dependent audibility,
        spectral smearing, and altered effective dynamic range**.

        It is deliberately labelled a prototype because this internal DSP engine
        has **not been clinically validated**.

        A research/clinical-development version should replace or benchmark this
        approximation against a validated hearing-loss model such as Cambridge
        MSBG/Clarity and should be tested with audiologists and listeners with
        measured hearing loss before making stronger perceptual claims.
        """
    )

st.divider()
st.caption(
    "Prototype V1 · Do not upload names, dates of birth, patient numbers, "
    "or other identifying health information."
)
