from datasets import load_dataset

def load_and_save_dataset():
  dataset = load_dataset("abisee/cnn_dailymail", "1.0.0")
  return dataset

def save_dataset(dataset, path):
  dataset.save_to_disk(path)
  
if __name__ == "__main__":
  dataset = load_and_save_dataset()
  save_dataset(dataset, '/content/Project1/data/processed')
