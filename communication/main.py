from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import logging


logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

CORS(app)
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=3003)