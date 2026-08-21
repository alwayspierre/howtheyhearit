# How Might They Hear It? — Web Prototype

A shareable Streamlit version of the audiogram-driven hearing simulation prototype.

## What it does

- Type a sentence and generate speech
- Record a sentence in the browser
- Upload an audio file
- Enter left- and right-ear audiogram thresholds
- Adjust assumed speech level
- Listen to normal vs simulated hearing
- Download the simulated stereo WAV

## Important scientific note

This is an educational prototype, **not a diagnostic or clinical tool**.

The current DSP engine approximates:

- frequency-dependent audibility loss
- spectral smearing
- altered effective dynamic range / recruitment

It is not yet a clinically validated model of an individual's subjective hearing.

Before clinical or commercial claims are made, the hearing engine should be benchmarked or replaced with a validated model such as Cambridge MSBG/Clarity and tested with audiologists and listeners with measured hearing loss.

## Privacy note

The prototype deliberately avoids accounts and databases.

- Recorded/uploaded audio is processed in the running app session.
- Typed text uses Google text-to-speech (gTTS), so the typed sentence is sent to Google's speech service.
- Do **not** enter names, dates of birth, patient numbers, or identifying health information.

For a production healthcare deployment, data protection, consent, retention, hosting, access control, and applicable health/privacy law should be designed properly.

---

# Deploy it to Streamlit Community Cloud

## 1. Create a GitHub repository

Create a new repo, for example:

`how-might-they-hear`

Upload the complete contents of this folder to the repository root:

- `app.py`
- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`
- `README.md`

## 2. Open Streamlit Community Cloud

Go to:

https://share.streamlit.io

Sign in with GitHub.

## 3. Create the app

Choose:

- Repository: your new GitHub repository
- Branch: `main`
- Main file path: `app.py`

Choose your preferred app URL if Streamlit offers it.

Then deploy.

You should receive a public address similar to:

`https://how-might-they-hear.streamlit.app`

## 4. Updating the app

Edit the GitHub files and commit/push them. Streamlit Community Cloud will redeploy from the repository.

---

# Run locally

You need Python and ffmpeg.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

For local use, ffmpeg must also be installed separately.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

---

# Suggested next version

1. Upload and visually parse an audiogram.
2. Ask the user to confirm extracted values before simulation.
3. Add optional background-noise presets (quiet room / classroom / restaurant).
4. Replace or benchmark the current simulator against Cambridge MSBG/Clarity.
5. Add an audiologist-facing validation mode.
6. Only after validation, consider saved patient profiles or authenticated clinical workflows.
