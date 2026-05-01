from model_wrapper import MyModel
import numpy as np

model = MyModel()

test_tweets = [
    "I'm so proud and happy about this!",
    "This makes me angry and disgusted.",
    "I feel scared but hopeful.",
]

output = model.predict(test_tweets)

print("Output shape:", output.shape)
print("Output:\n", output)
print("All Good:\n",output)