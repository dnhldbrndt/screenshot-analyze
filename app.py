from flask import Flask
import analyzer

app = Flask(__name__)

@app.route('/')
def home():
    return "Ready."

@app.route('/test')
def test():    
    return "Test."

@app.route('/screenshot', methods=['POST'])
def hello():
    analyzeimage()
    
 
if __name__ == "__main__":
    app.run(debug=True)