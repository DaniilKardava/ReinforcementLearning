import json
import numpy as np

load_path = "sb_weights/win_30_model_weights_og.json"

with open(load_path, "r") as f:
        params = json.load(f)

for key, value in params.items():
    params[key] = np.array(value)

items = list(params.items())

# Switch last value function weights and biases with policy function weights and biases:
items[-1], items[-2], items[-3], items[-4] = items[-3], items[-4], items[-1], items[-2]

params = dict(items)
keys = list(params.keys())

# Set proper gate order in gate matrices.
for i in range(4,7):

    data  = params[keys[i]]

    if i == 6:
        data = np.expand_dims(data, axis = 0)

    matrices = np.split(data, 4, axis=1)
    matrices[-1], matrices[-2] = matrices[-2], matrices[-1]
    data = np.concatenate(matrices, axis=1)

    if i == 6:
        data = np.squeeze(data)
    
    params[keys[i]] = data

# Save, only one time to not get lost with weights
save_path = "sb_weights/win_30_model_weights_reordered.json"

for key, value in params.items():
    params[key] = value.tolist()

with open(save_path, "w") as f:
    json.dump(params, f, indent=4)