import json
import os

# Load the train data
def load_train_data():
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, '..', 'data', 'trains.json')
    
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    print(f"Loaded {len(data)} trains")
    return data

if __name__ == "__main__":
    trains = load_train_data()
    print("First train:", trains[0]['id'])