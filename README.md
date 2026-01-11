# M2AI Afective Computing (AC) Project

---

## Datasets Used

- [Speech Emotion Recognition Voice Dataset](https://www.kaggle.com/datasets/tapakah68/emotions-on-audio-dataset)
> Audio files labeled with emotions, varing in gender, age and country of origin.

- [Steam Reviews](https://www.kaggle.com/datasets/andrewmvd/steam-reviews)
> Text reviews from Steam users.


- Demo Phrases
> A set of predefined phrases used for testing.

### 1. Download Datasets

```bash
curl -L -o ./datasets/steam-reviews.zip https://www.kaggle.com/api/v1/datasets/download/andrewmvd/steam-reviews && \
curl -L -o ./datasets/emotions-on-audio-dataset.zip https://www.kaggle.com/api/v1/datasets/download/tapakah68/emotions-on-audio-dataset
```

### 2. Unzip Datasets

```bash
unzip ./datasets/steam-reviews.zip -d ./datasets/text && \
unzip ./datasets/emotions-on-audio-dataset.zip -d ./datasets/audio
```

### 3. Cleanup

```bash
rm ./datasets/steam-reviews.zip && \
rm ./datasets/emotions-on-audio-dataset.zip
```

---

## Environment Setup

### 1. Install Dependencies

Using conda:
```bash
conda env create -f environment.yml
conda activate m2ai-ac
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 2. Download Language Models
You must download the English language model for the NLP analysis to work:

```bash
python -m spacy download en_core_web_sm
```

> Note: The first time you run the project, it will also download the Transformer models (Whisper/RoBERTa) from HuggingFace. This may take a few minutes depending on your internet connection.

---

## Running the Project

After activating the environment, run:

```bash
python src/main.py run configs/default.json
```

To re print results, use:

```bash
python src/main.py load results/demo_1768151117802.json
```

---

## Configuration

The project configuration is defined in:

```
configs/default.json
```
