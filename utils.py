def log_message(message):
    print(f"[LOG] {message}")

def load_config(config_file):
    import json
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def visualize_data(data, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(data)
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()