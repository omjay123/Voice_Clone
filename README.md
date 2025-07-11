# 🎙 Voice Cloning Prototype – India Speaks IVR

This is a quick prototype for a next-gen IVR system that greets callers using the *cloned voice of their dedicated account manager*, trained on just a few reference mel-spectrograms.

## 🚀 Objective

Build a simple *multi-speaker voice cloning model* that:
- Takes a reference mel-spectrogram input
- Encodes speaker identity via a lightweight speaker encoder
- Decodes a new mel-spectrogram using the speaker embedding

> 📢 Final output is a cloned mel-spectrogram. Vocoding (waveform synthesis) is assumed to be handled downstream (e.g., via HiFi-GAN).

---

## 🧱 Project Structure

```

voice-cloning/
├── model.py                  # Speaker encoder + mel decoder
├── data_loader.py           # Mel CSV loader (80x50 -> 4000 floats)
├── train.py                 # Trains model and saves training\_curve.png
├── infer.py                 # Inference from reference mels
├── requirements.txt         # Dependencies
├── README.md                # Project info
├── mel_train.csv            # Training set
├── mel_val.csv              # Validation set
├── mel_reference.csv        # Input for cloning
├── cloned_mel_predictions.csv  Output predictions
├── training_curve.png        Training loss plot
├── design_brief.pdf          Architecture + Results

```

---

## 🛠 Setup Instructions

### 1. Clone and Setup
```bash
git clone https://github.com/omjay123/Voice_Clone.git
cd voice-cloning
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 2. Train the Model

bash
python train.py


* Trains for 50 epochs
* Saves the model as model.pt
* Plots and saves training_curve.png

### 3. Run Inference

bash
python infer.py


* Reads mel_reference.csv
* Outputs cloned_mel_predictions.csv with predicted 4000-float mel-spectrograms

---

## 📊 Output Example (cloned_mel_predictions.csv)

csv
speaker_id,predicted_mel_flat
0,0.0023 0.0051 0.0079 ... (4000 floats)
1,...


Each row contains a reconstructed mel spectrogram conditioned on the speaker's voice.

---

## 📄 Included: desin_brief.pdf

Covers:

* Model architecture (encoder + decoder)
* Training results + loss plot
* Improvement roadmap (e.g., HiFi-GAN, Tacotron, larger datasets)

---

## 📈 Performance

* Final train loss: \~0.018
* Final val loss: \~0.020
* Model overfits the small dataset, as intended for proof-of-concept

---

## 🧠 Future Roadmap

* Train on larger multi-speaker datasets (e.g., VCTK)
* Integrate HiFi-GAN or Bark for waveform synthesis
* Add Streamlit web demo or REST API for IVR integration
