from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer

def load_dataset(path):
  return load_from_disk(path)

def tokenize_data(examples, tokenizer):
  inputs = tokenizer(examples['article'], max_length=512, truncation=True, padding='max_length')
  outputs = tokenizer(examples['highlights'], max_length=512, truncation=True, padding='max_length')
  return {
    'input_ids': inputs['input_ids'],
    'attention_mask': inputs['attention_mask'],
    'labels': outputs['input_ids']
  }

def preprocess_dataset(dataset, tokenizer):
  tokenized_train = dataset['train'].map(lambda x: tokenize_data(x, tokenizer), batched=True, remove_columns=['article', 'highlights'])
  tokenized_val = dataset['validation'].map(lambda x: tokenize_data(x, tokenizer), batched=True, remove_columns=['article', 'highlights'])
  tokenized_test = dataset['test'].map(lambda x: tokenize_data(x, tokenizer), batched=True, remove_columns=['article', 'highlights'])
  return tokenized_train, tokenized_val, tokenized_test

if __name__ == "__main__":
  dataset = load_dataset('/content/Project1/data/raw/cnn_dailymail')
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenized_train, tokenized_val, tokenized_test = preprocess_dataset(dataset, tokenizer)
  tokenized_train.save_to_disk('/content/Project1/data/processed/tokenized_data_train')
  tokenized_val.save_to_disk('/content/Project1/data/processed/tokenized_data_val')
  tokenized_test.save_to_disk('/content/Project1/data/processed/tokenized_data_test')
