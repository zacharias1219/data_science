from flask import Flask, request
import pickle

app = Flask(__name__)

@app.route('/', methods=['POST'])
def save_data():
    data = request.get_json()
    with open('data.pickle', 'wb') as file:
        pickle.dump(data, file)
    return 'Data saved successfully!'

@app.route('/', methods=['GET'])
def load_data():
    try:
        with open('data.pickle', 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        return 'No data found.'

if __name__ == '__main__':
    app.run()