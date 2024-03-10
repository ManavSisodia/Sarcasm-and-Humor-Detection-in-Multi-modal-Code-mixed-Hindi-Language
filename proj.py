import tensorflow as tf

model=tf.saved_model.load("D:/projects/Project+survey/Project/Project/15khumour+sarasmmodel2.tf")

signature = model.signatures['default']

input_tensor = signature['inputs']['input']
output_tensor = signature['outputs']['output']

user_input = input()

predictions = model(input_tensor=user_input)

output = predictions['output']

print(output)

