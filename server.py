'''Deploy a Flask application that will allow a user to provide
a text string which will then be analyzed to determine which emotion amongst a set of 6
is the most likely emotion being conveyed by the given text.
'''
from flask import Flask, request, render_template, jsonify
from EmotionDetection.emotion_detection import (emotion_detector, model_info,
                                                model_ready, MODEL_MISSING_MESSAGE)
import monetization

app = Flask(__name__, static_folder='public', static_url_path='')
app.json.sort_keys = False  # keep the emotions in the model's order
monetization.init_app(app)


def _read_text():
    '''Get the text to analyze from the query string, a form, or a JSON body.'''
    body = request.get_json(silent=True)
    if isinstance(body, dict) and 'textToAnalyze' in body:
        return body['textToAnalyze']
    return request.values.get('textToAnalyze')


def _reject(wants_json, message, status, **extra):
    '''Explain why a request was refused: JSON for the page, plain text otherwise.'''
    if wants_json:
        return jsonify(error=message, **extra), status
    return message


@app.route("/emotionDetector", methods=["GET", "POST"])
def emotion_analyzer():
    '''Retrieve the provided text string from the user, then pass the text
    to be analyzed by the emotion detector. Finally, return a response displaying
    the confidence scores across all emotions and the dominant emotion.

    Add ?format=json to get the scores as JSON (used by the web page).
    '''
    wants_json = request.values.get('format') == 'json'
    text_to_analyse = _read_text()
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return _reject(wants_json, "Invalid text! Please try again", 400)

    if not model_ready():
        return _reject(wants_json, MODEL_MISSING_MESSAGE, 503)

    status = monetization.get_status()
    if len(text_to_analyse) > status['max_chars']:
        message = (f"This text is {len(text_to_analyse):,} characters long. "
                   f"Your plan allows up to {status['max_chars']:,}.")
        return _reject(wants_json, message, 413, upgrade=status['tier'] == 'free')

    allowed, status = monetization.consume_analysis()
    if not allowed:
        message = ("You have used all your free analyses for today. "
                   "Upgrade to Pro for unlimited use, or come back tomorrow.")
        return _reject(wants_json, message, 402, upgrade=True)

    emotion_result = emotion_detector(text_to_analyse)
    dominant_emotion = emotion_result.pop('dominant_emotion')

    if wants_json:
        return jsonify(scores=emotion_result, dominant_emotion=dominant_emotion,
                       status=status)

    scores = ', '.join(f"'{name}': {value}" for name, value in emotion_result.items())
    response_str = f"""For the given statement, the system response is
    {scores}.
    The dominant emotion is <strong>{dominant_emotion}</strong>."""
    return response_str


@app.route("/api/model")
def model_details():
    '''Describe the trained model (size of training set, accuracy on unseen texts).'''
    info = model_info()
    if info is None:
        return jsonify(error=MODEL_MISSING_MESSAGE), 503
    return jsonify(info)


@app.route("/")
def render_index_page():
    '''Render the index page to the user, this is where the text string to be
    analyzed is provided and a response is displayed back to the user.
    '''
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
