from flask import Flask, render_template, request
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_data = None
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            blob = TextBlob(text)
            sentiment_data = {
                'text': text,
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
    
    return render_template('index.html', sentiment_data=sentiment_data)

if __name__ == '__main__':
    app.run(debug=True)
