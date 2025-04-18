import os
from flask import Flask, request, jsonify
import redis

app = Flask(__name__)

REDIS_HOST = os.environ.get('REDIS_HOST', 'redis')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

@app.route('/set', methods=['POST'])
def set_value():
    data = request.get_json()
    key = data.get('key')
    value = data.get('value')
    if not key or value is None:
        return jsonify({'error': 'Missing key or value'}), 400
    r.set(key, value)
    return jsonify({'message': f'Set {key} = {value}'}), 200

@app.route('/get/<key>', methods=['GET'])
def get_value(key):
    value = r.get(key)
    if value is None:
        return jsonify({'error': 'Key not found'}), 404
    return jsonify({'key': key, 'value': value}), 200

@app.route('/health', methods=['GET'])
def health():
    try:
        pong = r.ping()
        if pong:
            return jsonify({'status': 'ok'}), 200
    except Exception as e:
        return jsonify({'status': 'error', 'details': str(e)}), 500
    return jsonify({'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
