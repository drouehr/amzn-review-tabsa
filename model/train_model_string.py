from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, DistilBertForTokenClassification, DistilBertTokenizerFast
from sklearn.model_selection import train_test_split
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import transformers

# convert tag names to ids, sentiment 0 = neg, 1 = neu, 2 = pos
tag2id = {'O': 0, 'B-0': 1, 'I-0': 2, 'B-1': 3, 'I-1': 4, 'B-2': 5, 'I-2': 6}
id2tag = {0: 'O', 1: 'B-NEG', 2: 'I-NEG', 3: 'B-NEU', 4: 'I-NEU', 5: 'B-POS', 6: 'I-POS'}

class NotableAttributeDataset(Dataset):
  def __init__(self, reviews, labels, tokenizer):
    self.reviews = reviews
    self.labels = labels
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, idx):
    review = self.reviews[idx]['content']
    label_dict = next(l for l in self.labels if l['id'] == self.reviews[idx]['id'])
    
    # tokenize review text
    encodings = self.tokenizer(review, return_offsets_mapping=True, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    offsets = encodings['offset_mapping'].squeeze(0)
    input_ids = encodings['input_ids'].squeeze(0)
    
    # initialize all tags as 'O'
    ner_tags = ['O'] * input_ids.size(0)
    
    # adjust tags based on labeled attributes
    for attr, sentiment in label_dict['notableAttributes']:
      attr_start = review.find(attr)
      attr_end = attr_start + len(attr)

      # get start and end token positions for the attribute
      token_start_index, token_end_index = None, None
      for i, (start, end) in enumerate(offsets):
        if start <= attr_start < end:
          token_start_index = i
        if start < attr_end <= end:
          token_end_index = i
          break
      
      if token_start_index is not None and token_end_index is not None:
        # assign the B and I tags to the tokens corresponding to the attribute
        ner_tags[token_start_index] = f'B-{sentiment}'
        for j in range(token_start_index + 1, token_end_index + 1):
          ner_tags[j] = f'I-{sentiment}'

    for tag in ner_tags:
      if tag not in tag2id:
        raise ValueError(f"invalid tag '{tag}' found")

    labels = [tag2id[tag] for tag in ner_tags]
    encodings['labels'] = torch.tensor(labels)
    return {key: val.squeeze() for key, val in encodings.items()}

def compute_class_weights(reviews, labels, tokenizer):
  class_counts = {tag: 0 for tag in tag2id.keys()}
  total_tokens = 0

  for review, label_dict in zip(reviews, labels):
    review_text = review['content'] 
    encodings = tokenizer(review_text, return_offsets_mapping=True, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    offsets = encodings['offset_mapping'].squeeze(0)
    input_ids = encodings['input_ids'].squeeze(0)

    ner_tags = ['O'] * input_ids.size(0)

    for attr, sentiment in label_dict['notableAttributes']:
      attr_start = review_text.find(attr)
      attr_end = attr_start + len(attr)

      token_start_index, token_end_index = None, None
      for i, (start, end) in enumerate(offsets):
        if start <= attr_start < end:
          token_start_index = i
        if start < attr_end <= end:
          token_end_index = i
          break

      if token_start_index is not None and token_end_index is not None:
        ner_tags[token_start_index] = f'B-{sentiment}'
        for j in range(token_start_index + 1, token_end_index + 1):
          ner_tags[j] = f'I-{sentiment}'

    for tag in ner_tags:
      if tag != 'O':
        class_counts[tag] += 1
      total_tokens += 1

  # logarithmic dampening on minority classes, 1 for outside tag
  class_weights = {cls: np.log(total_tokens/(len(class_counts)*count)) if cls != 'O' else 0.2 for cls, count in class_counts.items()}
  print(f"class_weights = {class_weights}")
  weights = [class_weights[tag] if tag in class_counts else 0 for tag, idx in sorted(tag2id.items(), key=lambda item: item[1])]
  class_weights_tensor = torch.tensor(weights, dtype=torch.float)
  print(f"class_weights_tensor = {class_weights_tensor}")

  return class_weights_tensor


def collate_fn(batch):
  input_ids = [item['input_ids'] for item in batch]
  attention_mask = [item['attention_mask'] for item in batch]
  labels = [item['labels'] for item in batch]
  
  input_ids = pad_sequence(input_ids, batch_first=True)
  attention_mask = pad_sequence(attention_mask, batch_first=True)
  labels = pad_sequence(labels, batch_first=True)

  return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

def align_predictions(preds, label_ids, attention_mask):
  aligned_preds = []
  aligned_labels = []

  for i in range(attention_mask.shape[0]):
    valid_indexes = attention_mask[i].bool()
    batch_pred_ids = preds[i][valid_indexes].argmax(-1)
    batch_label_ids = label_ids[i][valid_indexes]
    batch_preds = [id2tag[pred_id] for pred_id in batch_pred_ids]
    batch_labels = [id2tag[label_id] for label_id in batch_label_ids]
    aligned_preds.append(batch_preds)
    aligned_labels.append(batch_labels)

  return aligned_preds, aligned_labels

print(f"environment:\n- PyTorch {torch.__version__}\n- CUDA {torch.version.cuda}\n- transformers {transformers.__version__}")

# load the reviews and label data
with open('reviews.json') as f:  
  reviews = json.load(f)
with open('labels.json') as f:
  labels = json.load(f)

print(f"loaded {len(reviews)} reviews and {len(labels)} labels")
print("sample review:", reviews[0])
print("sample label:", labels[0])

# filter out any reviews with empty content
filtered_reviews = [r for r in reviews if r['content'].strip()]
# create a set of review and label IDs from filtered_reviews and labels
review_ids = {r['id'] for r in filtered_reviews}
label_ids = {l['id'] for l in labels}
# find the intersection of review and label IDs
common_ids = review_ids & label_ids
# filter reviews and labels to only those with a common ID
filtered_reviews = [r for r in filtered_reviews if r['id'] in common_ids]
filtered_labels = [l for l in labels if l['id'] in common_ids]
assert len(filtered_reviews) == len(filtered_labels), f"number of filtered reviews ({len(filtered_reviews)}) does not match number of filtered labels ({len(filtered_labels)})"
print(f"> proceeding with {len(filtered_reviews)} review entries after filtering.")

model_dir = './string_model_distilbert' 
if not os.path.exists(model_dir):
  os.makedirs(model_dir, exist_ok=True)

# early stopping params
best_val_f1 = 0
epochs_no_improve = 0
patience = 3

train_reviews, val_reviews, train_labels, val_labels = train_test_split(filtered_reviews, filtered_labels, test_size=0.1)
print(f"training, validation dataset size: {len(train_reviews)}, {len(val_reviews)}")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_dataset = NotableAttributeDataset(train_reviews, train_labels, tokenizer) 
val_dataset = NotableAttributeDataset(val_reviews, val_labels, tokenizer)

assert torch.cuda.is_available(), f"CUDA is not available; check env"
device = torch.device('cuda')

class_weights_tensor = compute_class_weights(filtered_reviews, filtered_labels, tokenizer)

# move the tensor to the device
class_weights_tensor = class_weights_tensor.to(device)
# mod celoss instantiation to include class weights
loss_fn = CrossEntropyLoss(weight=class_weights_tensor)

model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=7)
model.to(device)

start_time = time.time()

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=8e-6)
total_steps = len(train_loader) * 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

max_epochs = 150
train_start_time = time.strftime("%Y%m%d_%H%M%S")

epoch_avg_train_accuracies, epoch_avg_train_precisions, epoch_avg_train_recalls, epoch_avg_train_f1s = [], [], [], []
epoch_avg_val_accuracies, epoch_avg_val_precisions, epoch_avg_val_recalls, epoch_avg_val_f1s = [], [], [], []

for epoch in range(max_epochs):
  print(f"epoch {epoch+1}/{max_epochs}")
  # tracking params
  batch_train_accuracies, batch_train_precisions, batch_train_recalls, batch_train_f1s = [], [], [], []
  batch_val_accuracies, batch_val_precisions, batch_val_recalls, batch_val_f1s = [], [], [], []
  
  # training stage
  model.train()
  print("# train")
  for step, batch in enumerate(train_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    logits = outputs.logits
    active_loss = attention_mask == 1  # only consider active parts of the loss

    active_logits = logits[active_loss]
    active_labels = labels[active_loss]
    loss = loss_fn(active_logits, active_labels)

    logits = outputs.logits.detach().cpu().numpy()
    label_ids = labels.detach().cpu().numpy() 

    attention_mask = attention_mask.cpu()
    train_preds, train_true = align_predictions(logits, label_ids, attention_mask)

    if step == 0:
      print(f"- train_preds[0] = {train_preds[0]}")
      print(f"- train_true[0]  = {train_true[0]}")

    batch_accuracy = accuracy_score(train_true, train_preds)
    batch_precision = precision_score(train_true, train_preds, average='weighted', scheme=IOB2)
    batch_recall = recall_score(train_true, train_preds, average='weighted', scheme=IOB2)
    batch_f1 = f1_score(train_true, train_preds, average='weighted', scheme=IOB2)

    batch_train_accuracies.append(batch_accuracy)
    batch_train_precisions.append(batch_precision)
    batch_train_recalls.append(batch_recall)
    batch_train_f1s.append(batch_f1)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if((step+1) % 20 == 0):
      print(f"  step {step + 1}/{len(train_loader)} - loss: {outputs.loss.item():.4f} - accuracy: {batch_accuracy} - precision: {batch_precision} - recall: {batch_recall} - f1: {batch_f1}")

  epoch_avg_train_accuracies.append(np.mean(batch_train_accuracies))
  epoch_avg_train_precisions.append(np.mean(batch_train_precisions))
  epoch_avg_train_recalls.append(np.mean(batch_train_recalls))
  epoch_avg_train_f1s.append(np.mean(batch_train_f1s))


  # validation stage
  model.eval()
  print("# validation")
  for step, batch in enumerate(val_loader):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    # apply attention mask
    labels = labels * attention_mask

    with torch.no_grad():  
      outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    
    logits = outputs.logits
    active_loss = attention_mask == 1  # only consider active parts of the loss
    active_logits = logits[active_loss]
    active_labels = labels[active_loss]
    loss = loss_fn(active_logits, active_labels)
    
    val_logits = outputs.logits.detach().cpu().numpy()
    val_label_ids = labels.detach().cpu().numpy()

    attention_mask = attention_mask.cpu()
    val_preds, val_true = align_predictions(val_logits, val_label_ids, attention_mask)

    if(step == 0):
      print(f" - val_preds[0] = {val_preds[0]}")
      print(f" - val_true[0] = {val_true[0]}")

    batch_accuracy = accuracy_score(val_true, val_preds)
    batch_precision = precision_score(val_true, val_preds, average='weighted', scheme=IOB2)
    batch_recall = recall_score(val_true, val_preds, average='weighted', scheme=IOB2)
    batch_f1 = f1_score(val_true, val_preds, average='weighted', scheme=IOB2)

    # store the metrics
    batch_val_accuracies.append(batch_accuracy)
    batch_val_precisions.append(batch_precision)
    batch_val_recalls.append(batch_recall)
    batch_val_f1s.append(batch_f1)

    if (step + 1) % 5 == 0:
      print(f"  step {step + 1}/{len(val_loader)} - loss: {outputs.loss.item():.4f} - accuracy: {batch_accuracy} - precision: {batch_precision} - recall: {batch_recall} - f1: {batch_f1}")

  # validation metrics over entire epoch using lists of lists (sequences)
  epoch_avg_val_accuracies.append(np.mean(batch_val_accuracies))
  epoch_avg_val_precisions.append(np.mean(batch_val_precisions))
  epoch_avg_val_recalls.append(np.mean(batch_val_recalls))
  epoch_avg_val_f1s.append(np.mean(batch_val_f1s))

  print(f"epoch {epoch + 1} <train> accuracy: {np.mean(batch_train_accuracies):.4f}, precision: {np.mean(batch_train_precisions):.4f}, recall: {np.mean(batch_train_recalls):.4f}, F1: {np.mean(batch_train_f1s):.4f}")
  print(f"epoch {epoch + 1} <validation> accuracy: {np.mean(batch_val_accuracies):.4f}, precision: {np.mean(batch_val_precisions):.4f}, recall: {np.mean(batch_val_recalls):.4f}, F1: {np.mean(batch_val_f1s):.4f}")


  # checkpointing starting from epoch>5
  if(epoch > 5):
    model_path = f'{model_dir}/{train_start_time}/epoch{epoch+1}' 
    model.save_pretrained(model_path)
    print(f"saved fine-tuned token classification model checkpoint to '{model_path}'")

    epochs = range(1, epoch+2)
    plt.figure(figsize=(15, 10))
    # training metrics
    plt.subplot(2, 2, 1)
    plt.plot(epochs, epoch_avg_train_accuracies, label='train accuracy')
    plt.plot(epochs, epoch_avg_val_accuracies, label='validation accuracy')
    plt.title('accuracy over epochs')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, epoch_avg_train_precisions, label='train precision')
    plt.plot(epochs, epoch_avg_val_precisions, label='validation precision')
    plt.title('precision over epochs')
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, epoch_avg_train_recalls, label='train recall')
    plt.plot(epochs, epoch_avg_val_recalls, label='validation recall')
    plt.title('recall over epochs')
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, epoch_avg_train_f1s, label='train F1 score')
    plt.plot(epochs, epoch_avg_val_f1s, label='validation F1 score')
    plt.title('F1 score over epochs')
    plt.xlabel('epoch')
    plt.ylabel('F1 score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_path, 'metrics_over_epochs.png'))

  scheduler.step()

  # early stopping if f1 has stagnated
  if np.mean(batch_val_f1s) > best_val_f1:
    best_val_f1 = np.mean(batch_val_f1s)
    epochs_no_improve = 0
  elif np.mean(batch_val_f1s) > 0.7:
    epochs_no_improve += 1
  if epochs_no_improve >= patience:
    print(f"stopping early at epoch {epoch+1} - no improvement after {patience} epochs")
    break

end_time = time.time()
total_time = end_time - start_time
print(f"training took {total_time:.2f} seconds")
