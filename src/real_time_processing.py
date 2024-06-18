import socket
import json
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def start_server(model):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 9999))
    server_socket.listen(1)
    print('Waiting for a connection...')
    connection, address = server_socket.accept()
    print('Connected by', address)

    while True:
        data = connection.recv(1024)
        if not data:
            break
        player_data = json.loads(data.decode('utf-8'))
        process_real_time_data(player_data, model)

def process_real_time_data(player_data, model):
    inputs = tokenizer(player_data, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(inputs.input_ids)
    challenging_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(challenging_output)
