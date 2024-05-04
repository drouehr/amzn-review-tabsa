from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
from transformers import DistilBertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import transformers

class DistilBertForMultiLabelSequenceClassification(DistilBertForSequenceClassification):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = 10
    self.classifier = torch.nn.Linear(config.dim, self.num_labels)

  def forward(self, input_ids, attention_mask=None, head_mask=None, labels=None, attr_weights=None):
    outputs = self.distilbert(input_ids, attention_mask=attention_mask, head_mask=head_mask)
    pooled_output = outputs[0][:, 0]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if attr_weights is None:
        # default to equal weights if not provided
        attr_weights = torch.ones(self.num_labels, device=labels.device)

      # normalize weights
      attr_weights = attr_weights / attr_weights.sum()
      # compute weighted MSE loss
      squared_errors = (logits - labels.float()) ** 2
      weighted_squared_errors = attr_weights.view(1, -1).expand(squared_errors.size(0), -1) * squared_errors
      loss = weighted_squared_errors.mean()

    return {'loss': loss, 'logits': logits}

class MultiAttrDataset(Dataset):
  def __init__(self, reviews, labels, attributes, tokenizer):
    self.reviews = reviews
    self.labels = labels
    self.attributes = attributes
    self.tokenizer = tokenizer
    assert len(self.reviews) == len(self.labels), f"mismatch between number of reviews ({len(self.reviews)}) and labels ({len(self.labels)})"

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, idx):

    review = self.reviews[idx]['content']
    label_dict = next((item for item in self.labels if item['id'] == self.reviews[idx]['id']), None)
    if label_dict is None:
      raise ValueError(f"no matching label found for review ID {self.reviews[idx]['id']}")
    labels = [label_dict[attr] for attr in self.attributes]
    encoding = self.tokenizer(review, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    return {
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(labels, dtype=torch.float)
    }
  
# custom collate function
def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  labels = torch.stack([item['labels'] for item in batch]) 
  input_ids = pad_sequence(input_ids, batch_first=True)
  attention_mask = pad_sequence(attention_mask, batch_first=True)

  return {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'labels': labels
  }

def calculate_class_weights(labels):
  weight_calc_starttime = time.time()
  label_counts = defaultdict(lambda: np.zeros(6))
  sentiment_counts = np.zeros(5)  # separate array for sentiment counts

  # iterate over the dataset to count occurrences of each label value
  for entry in labels:
    for key in entry:
      if key == 'sentiment':
        value = entry[key] - 1  # shift sentiment range from 1-5 to 0-4
        sentiment_counts[value] += 1
      elif key != 'id' and key != 'notableAttributes':  # exclude non-numeric attributes
        value = entry[key]
        label_counts[key][value] += 1

  # calculate the weights inversely proportional to the log of label counts
  class_weights = {}

  # handle sentiment separately
  sentiment_smoothed_log_counts = np.log(sentiment_counts + 1)
  sentiment_inv_freq = 1.0 / sentiment_smoothed_log_counts
  sentiment_weights = sentiment_inv_freq / sentiment_inv_freq.sum() * 5
  class_weights['sentiment'] = sentiment_weights

  for attr, counts in label_counts.items():
    # apply smoothing and take log to dampen the effect of large counts
    smoothed_log_counts = np.log(counts + 1)
    inv_freq = 1.0 / smoothed_log_counts
    weights = inv_freq / inv_freq.sum() * 6
    class_weights[attr] = weights

  print(f"class weights calculated in {time.time() - weight_calc_starttime}s: {class_weights}")
  return class_weights

print(f"environment:\n- PyTorch {torch.__version__}\n- CUDA {torch.version.cuda}\n- transformers {transformers.__version__}")

# load the reviews and label data
with open('Amazon-Reviews-2023-cuttolabellength.json') as f:  
  reviews = json.load(f)

with open('labels-exact.json') as f:
  labels = json.load(f)

print(f"loaded {len(reviews)} reviews and {len(labels)} labels")
print("sample review:", reviews[0])
print("sample label:", labels[0])

# filter out any reviews with empty content
filtered_reviews = [r for r in reviews if r['content'].strip()]
# create a set of review IDs from filtered_reviews
review_ids = {r['id'] for r in filtered_reviews}
# create a set of label IDs 
label_ids = {l['id'] for l in labels}
# find the intersection of review and label IDs
common_ids = review_ids & label_ids
# filter reviews and labels to only those with a common ID
filtered_reviews = [r for r in filtered_reviews if r['id'] in common_ids]
filtered_labels = [l for l in labels if l['id'] in common_ids]
assert len(filtered_reviews) == len(filtered_labels), f"number of filtered reviews ({len(filtered_reviews)}) does not match number of filtered labels ({len(filtered_labels)})"
print(f"> proceeding with {len(filtered_reviews)} review entries after filtering.")

# define the attributes to train the model for  
attributes = ['sentiment', 'quality', 'userExperience', 'usability', 'design', 'durability', 'pricing', 'asAdvertised', 'customerSupport', 'repurchaseIntent']
num_labels = len(attributes)
config = DistilBertConfig.from_pretrained('distilbert-base-uncased')
config.hidden_dropout_prob = 0.15 # default 0.1
config.num_labels = num_labels

# per-epoch losses for plotting
epoch_train_losses = []
epoch_train_mses = {attr: [] for attr in attributes}
epoch_val_losses = []
epoch_val_mses = {attr: [] for attr in attributes}

model_dir = './multi_attr_model_distilbert'
if not os.path.exists(model_dir):
  os.makedirs(model_dir, exist_ok=True)

# early stopping params
best_val_loss = float('inf')
best_epoch = 0
epochs_no_improve = 0
patience = 3

# load tokenizer 
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
dataset = MultiAttrDataset(filtered_reviews, filtered_labels, attributes, tokenizer)

# 10% validation set splitting
train_dataset, val_dataset = train_test_split(dataset, test_size=0.1, random_state=42)
print(f"training, validation dataset size: {len(train_dataset)}, {len(val_dataset)}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)  
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

assert torch.cuda.is_available(), f"CUDA is not available; check env"
device = torch.device('cuda')
print(f"using device: {device}")

class_weights = calculate_class_weights(filtered_labels)
class_weights = {attr: weights / weights.sum() for attr, weights in class_weights.items()}
# convert class weights to tensors and move to the device
class_weight_tensors = {}
for attr, weights in class_weights.items():
  class_weight_tensors[attr] = torch.tensor(weights, dtype=torch.float32).to(device)

# create the multi-attribute model, optimizer, lr scheduler 
model = DistilBertForMultiLabelSequenceClassification(config)
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3, verbose=True)
start_time = time.time()

max_epochs=50

# training loop
for epoch in range(max_epochs):
  print(f"epoch {epoch + 1}/{max_epochs}:")
  model.train()
   
  # training loop
  train_loss = 0.0
  train_mses = [0.0] * num_labels
  print("# train")

  for step, batch in enumerate(train_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device) 

    # extract attribute weights from class_weight_tensors
    attr_weights_tensor = torch.tensor([class_weights[attr][0] for attr in attributes], device=device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, attr_weights=attr_weights_tensor)
    loss = outputs['loss']
    logits = outputs['logits']

    labels_np = labels.cpu().detach().numpy()
    preds = logits.cpu().detach().numpy()
    mse_scores = [mean_squared_error(labels_np[:, i], preds[:, i]) for i in range(num_labels)]

    train_loss += loss.item()
    train_mses = [train_mses[i] + mse_scores[i] for i in range(num_labels)]

    loss.backward()
    optimizer.step() 
    model.zero_grad()

    if (step + 1) % 10 == 0:
      formatted_mse_scores = [f"{score:.4f}" for score in mse_scores]
      print(f"  step {step + 1}/{len(train_loader)} - loss: {loss.item():.4f} - mses: {formatted_mse_scores}")

  avg_train_loss = train_loss / len(train_loader)
  avg_train_mses = [mse / len(train_loader) for mse in train_mses]
  epoch_train_losses.append(avg_train_loss)
  
  for i, attr in enumerate(attributes):
    epoch_train_mses[attr].append(avg_train_mses[i])
  print(f"> epoch {epoch + 1} <TRAIN> avg loss: {avg_train_loss:.4f} - avg mse: {avg_train_mses}")
  
  # validation loop
  print("# validation")
  model.eval()
  val_loss = 0.0
  val_mses = [0.0] * num_labels

  with torch.no_grad():
    for step, batch in enumerate(val_loader):
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device) 
      labels = batch['labels'].to(device)

      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
      loss = outputs['loss']
      logits = outputs['logits']

      labels_np = labels.cpu().detach().numpy()  
      preds = logits.cpu().detach().numpy()
      mse_scores = [mean_squared_error(labels_np[:, i], preds[:, i]) for i in range(num_labels)]

      val_loss += loss.item()
      val_mses = [val_mses[i] + mse_scores[i] for i in range(num_labels)]

      if (step + 1) % 10 == 0:
        formatted_mse_scores = [f"{score:.4f}" for score in mse_scores]
        print(f"  step {step + 1}/{len(val_loader)} - loss: {loss.item():.4f} - mses: {formatted_mse_scores}")

  avg_val_loss = val_loss / len(val_loader)  
  avg_val_mses = [mse / len(val_loader) for mse in val_mses]
  epoch_val_losses.append(avg_val_loss) 

  for i, attr in enumerate(attributes):
    epoch_val_mses[attr].append(avg_val_mses[i])
  print(f"> epoch {epoch + 1} <VAL> avg loss: {avg_val_loss:.4f} - avg mse: {avg_val_mses}")


  model_path = f'{model_dir}/{start_time}/epoch{epoch+1}' 
  model.save_pretrained(model_path)
  print(f"saved fine-tuned token classification model checkpoint to '{model_path}'")

  epochs = range(1, epoch+2)
  plt.figure(figsize=(15, 10))
  # training metrics
  fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(16, 15))
  num_executed_epochs = len(epoch_train_losses)
  epochs = range(1, num_executed_epochs+1)

  # training loss graph
  ax1.set_title('training, validation loss / epochs')
  ax1.set_xlabel('epoch')
  ax1.set_ylabel('loss')
  ax1.plot(epochs, epoch_train_losses, label='training')
  ax1.plot(epochs, epoch_val_losses, label='validation')
  ax1.legend()

  # training MSE graph
  ax2.set_title('training MSE / epochs')
  ax2.set_xlabel('epoch')
  ax2.set_ylabel('MSE')
  for attr, mses in epoch_train_mses.items():
    ax2.plot(epochs, mses, label=attr)
  ax2.legend()

  # validation MSE graph
  ax3.set_title('validation MSE / epochs')
  ax3.set_xlabel('epoch')
  ax3.set_ylabel('MSE')
  for attr, mses in epoch_val_mses.items():
    ax3.plot(epochs, mses, label=attr)
  ax3.legend()

  plt.tight_layout()
  plt.savefig(os.path.join(model_path, 'metrics_over_epochs.png'))

  # update lr scheduler
  scheduler.step(avg_val_loss)
  print(f"current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

  if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    best_epoch = epoch
    epochs_no_improve = 0
  else:
    epochs_no_improve += 1
    
end_time = time.time()
total_time = end_time - start_time
print(f"training took {total_time:.2f} seconds.\nlowest validation loss was {best_val_loss} achieved at epoch {best_epoch+1}")
