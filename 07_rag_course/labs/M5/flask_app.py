from flask import Flask, request, jsonify
import threading
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import torch
import threading
import logging
device = "cuda" if torch.cuda.is_available() else "cpu"
# Initialize models globally to load them once
model_name = 'models/BAAI/bge-base-en-v1.5'
model = SentenceTransformer(model_name).to(device) # Ensure "cuda" is available, otherwise use "cpu"
app = Flask(__name__)

@app.route('/.well-known/ready', methods=['GET'])
def readiness_check():
    return "Ready", 200

@app.route('/meta', methods=['GET'])
def readiness_check_2():
    return jsonify({'status': 'Ready'}), 200


@app.route('/vectors', methods=['POST']) 
def vectorize():
    try:
        try:
            data = request.json.get('text')
        except Exception as e:
            try:
                data = request.data.decode("utf-8")
            except Exception as e:
                print(e)
        text = json.loads(data)
        if isinstance(text, str):
            text = [text]
        else:
            text =text['text']
            
        embeddings = model.encode(text, convert_to_tensor=True)
        embeddings_list = embeddings.cpu().numpy().tolist()

        return jsonify({'vector': embeddings_list[0] if len(embeddings_list) == 1 else embeddings_list})


    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
app.logger.disabled = True
# Get the Flask app's logger
log = logging.getLogger('werkzeug')
# Set logging level (ERROR or CRITICAL suppresses routing logs)
log.setLevel(logging.ERROR)
def run_app():
    app.run(host='0.0.0.0', port=5000, debug = False)

flask_thread = threading.Thread(target=run_app)
flask_thread.start()
