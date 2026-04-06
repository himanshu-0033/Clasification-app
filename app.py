from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    # Run the Flask app on default port 5000
    app.run(debug=True, port=5000)
