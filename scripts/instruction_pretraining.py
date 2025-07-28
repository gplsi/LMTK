import json
import os

dataset_path = "/workspace/data/v5/"
output_path = "/workspace/output/data/"

if not os.path.exists(output_path):
    os.makedirs(output_path)
# walk to find all the json files
json_files = []
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".json"):
            json_files.append(os.path.join(root, file))
            
print(f"Found {len(json_files)} JSON files in the dataset path.")

text = ""
for file in json_files:
    empty_keys = {}
    with open(file, 'r') as f:
        data = json.load(f)
    # For each instance in the JSON, check the available keys
    conversations = []
    index = 0
    for instance in data:
        # Check if the input and assistance keys are list
        if isinstance(instance.get('input'), list) and isinstance(instance.get('assistance'), list):
            # If they are lists, we can build the conversation iterating over each one, we need to iterate over the longer one
            system = instance['system']
            # TODO: add a logic to load the system prompt for the tokenizer of that model if the system found is empty
            input_list = instance['input']
            assistance_list = instance['assistance']
            target = instance['target']
            max_length = max(len(input_list), len(assistance_list))
            conversation = []
            for i in range(max_length):
                if i < len(input_list):
                    conversation.append(input_list[i])
                if i < len(assistance_list):
                    conversation.append(assistance_list[i])
            conversation.append(target)
            # Add the system prompt if it exists
            if system:
                conversation.insert(0, system)
            # Add all the list as a string
            conversation = "\n".join(conversation)
            # Save the conversation as a separate txt with the name of the original json and the index
            conversation_file_name = f"{os.path.splitext(os.path.basename(file))[0]}_{index}.txt"
            conversation_file_path = os.path.join(output_path, conversation_file_name)
            with open(conversation_file_path, 'w') as conv_file:
                conv_file.write(conversation)
            print(f"Saved conversation to {conversation_file_path}")
            index += 1

            #conversations.append(conversation)  
        else:
            raise NotImplementedError("Input and assistance must be lists.")
#     # Join all conversations with a newline
#     text += "\n".join(conversations) + "\n"
    
# # Save the text to a file
# output_file = os.path.join(dataset_path, "conversations.txt")
# with open(output_file, 'w') as f:
#     f.write(text)
# print(f"Conversations saved to {output_file}.")