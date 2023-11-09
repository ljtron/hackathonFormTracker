from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    return render_template('frontEnd/main.html')

@socketio.on('joined')
def handle_message(data):
    print("new user", data)
    emit('back', data, broadcast=True)

@socketio.on('message')
def handle_message(data):
    print("Data received from frontend:", data)
    # Process the data as needed


if True:
    print("Starting the server...")
    socketio.run(app, port=8001, debug=True)