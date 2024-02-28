from flask import Flask, render_template, request
from summarizer.sbert import SBertSummarizer

app = Flask(__name__)

# Function to generate summary
def generate_summary(body):
    model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
    summary = model(body, num_sentences=5)
    return summary

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Predict page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        body = request.form['data']
        if body:
            summary = generate_summary(body)
            return render_template('result.html', summary=summary)
        else:
            error_message = "Please enter some text to summarize."
            return render_template('predict.html', error_message=error_message)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
