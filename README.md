# PRODIGY_GA_01
Train a model to generate coherent and contextually relevant text based on a given prompt. Starting with GPT-2, a transformer model developed by OpenAI, you will learn how to fine-tune the model on a custom dataset to create text that mimics the style and structure of your training data. 

---

# Training a GPT-2 Model for Text Generation

In this tutorial, we will go through the process of training a GPT-2 model to generate coherent and contextually relevant text based on a given prompt. GPT-2, developed by OpenAI, is a powerful transformer model known for its ability to produce human-like text. By fine-tuning GPT-2 on a custom dataset, we can generate text that closely mimics the style and structure of our training data.

## Prerequisites

Ensure you have the following before starting:

- Python installed on your machine.
- PyTorch: The deep learning framework required for GPT-2.
- Transformers library from Hugging Face.
- A custom dataset: The text data on which we will fine-tune the model.

## Steps to Train the GPT-2 Model

### 1. Set Up the Environment

First, install the necessary libraries:  pip install torch transformers


### 2. Load the GPT-2 Model and Tokenizer

We begin by loading the pre-trained GPT-2 model and tokenizer using the transformers library:

from transformers import GPT2LMHeadModel, GPT2Tokenizer

Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


### 3. Prepare the Dataset
Assume you have a text file named custom_dataset.txt containing your training data. Load and preprocess this data:

with open('custom_dataset.txt', 'r', encoding='utf-8') as file:
    dataset = file.read()

Tokenize the dataset
inputs = tokenizer(dataset, return_tensors='pt', max_length=1024, truncation=True)


### 4. Fine-Tune the Model
Next, we define a custom dataset class and a training loop to fine-tune the model on your dataset:

from torch.utils.data import DataLoader, Dataset
import torch

Create a custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=1024):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True)
        inputs['labels'] = inputs.input_ids.clone()
        return inputs

Create a DataLoader
texts = dataset.split('\n')
train_dataset = TextDataset(texts, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

Set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

Training loop
model.train()
for epoch in range(3):  # number of epochs
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

        
### 5. Generate Text
After fine-tuning the model, you can generate text based on a prompt:

Set the model to evaluation mode
model.eval()

Define the prompt
prompt = "Once upon a time"

Tokenize the prompt
inputs = tokenizer(prompt, return_tensors='pt')

Generate text
outputs = model.generate(inputs.input_ids, max_length=100, num_return_sequences=1)

Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

## Conclusion
By following these steps, you can fine-tune GPT-2 to generate text that is coherent and contextually relevant to your custom dataset. Fine-tuning allows you to adapt the model to specific styles, structures, or topics, making it a powerful tool for various text generation tasks.

You can save this content in a file named `README.md` in your GitHub repo
