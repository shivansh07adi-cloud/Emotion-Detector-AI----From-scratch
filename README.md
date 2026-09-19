<h1 align="center"> IBM Full Stack Software Developer Certificate <br> Developing AI Applications with Python and Flask </h1>

## Emotion Detector

This is the final project within the course, "Developing AI Applications with Python and Flask" in the IBM Full Stack Software Developer Certificate. The objective of this project is to develop an AI-based Flask Web Application which will allow a user to provide a text string as input and receive a response from the AI which will tell the user what emotion is being conveyed in that text string.

More information about the course can be found [here](https://www.coursera.org/learn/python-project-for-ai-application-development/)

In this version the AI is **trained from scratch**: no API and no pretrained model. `train.py` learns from 16,000 labelled tweets with a hand-written softmax regression (TF-IDF features, Adam optimiser, L2 regularisation, early stopping) using only NumPy. The learned weights are saved in `EmotionDetection/model.json`, and the web app predicts with plain Python.

It detects six emotions: sadness, joy, love, anger, fear and surprise.

## Train the model (once)

```bash
pip install -r requirements-train.txt
python download_data.py     # saves data/train.csv, validation.csv, test.csv
python train.py             # writes EmotionDetection/model.json
```

`train.py` prints accuracy, macro F1, per-emotion precision and recall, and a confusion matrix on the test set (tweets the model never saw). That accuracy is also shown at the bottom of the web page.

Useful options:

- `--class-weight balanced` helps rare emotions such as surprise (usually costs a little overall accuracy).
- `--neutral-below 0.4` reports "No strong emotion" when the top probability is below this value. Raise or lower it after looking at the confidence numbers `train.py` prints.
- `--max-features`, `--epochs`, `--lr`, `--l2` are described by `python train.py --help`.

No `datasets` package? Download the `dair-ai/emotion` splits yourself and save them as `data/train.csv`, `data/validation.csv` and `data/test.csv` with the columns `text,label` (label as a word such as `joy`).

## Run it

```bash
pip install -r requirements.txt
python server.py          # http://localhost:5000
python -m unittest        # runs the tests (they train a tiny model on made-up sentences)
```

`/emotionDetector?textToAnalyze=...` returns the original text response. Add `&format=json` for JSON scores.

The model learned from tweets that say how the writer feels, such as "i feel so lonely", so it works best on first-person feeling statements. It has no "neutral" class, so plain factual sentences get a low-confidence answer.

## Deploy on Vercel (demo)

Vercel detects `server.py` and `requirements.txt` on its own, so there is no config file to write.

1. Train the model first, then commit `EmotionDetection/model.json` (about 1 MB). Do not commit `data/`.
2. Push the project to GitHub and import the repo in Vercel (or run `vercel` from this folder).
3. On Vercel the app runs in **demo mode**: no daily limit, no payments, and the Get Pro section is hidden, because Vercel has no persistent disk for the license database.

Static files live in `public/` because Vercel serves that folder from its CDN and ignores Flask's own static folder.

## Monetization (needs a server with a persistent disk)

Visitors get a free daily quota. A one-time **Pro pass** (paid with Razorpay Checkout, UPI and cards) removes the quota and raises the text limit. A successful payment creates a license key that unlocks Pro on any browser.

1. Copy `.env.example` to `.env` and fill in `SECRET_KEY` and your Razorpay **test** keys.
2. Load them into the environment and start the server. With no Razorpay keys the Get Pro button is disabled and everything else works.
3. Make a test payment, then switch to live keys.

Plan limits, price and pass length are environment variables (see `.env.example`). Usage, orders and licenses are stored in a local SQLite file (`emotion_detector.db`), which must live on a persistent disk when you deploy. Behind a proxy, set `TRUST_PROXY=1`. For production use a real WSGI server, for example `gunicorn -w 2 server:app`.
