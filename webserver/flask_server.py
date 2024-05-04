from flask import Flask, request, jsonify
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, DistilBertForTokenClassification
import time

def is_number(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

class DistilBertForMultiLabelSequenceClassification(DistilBertForSequenceClassification):
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = 10
    self.classifier = torch.nn.Linear(config.dim, self.num_labels)

  def forward(self, input_ids=None, attention_mask=None, head_mask=None, labels=None):
    outputs = self.distilbert(input_ids, attention_mask=attention_mask, head_mask=head_mask)
    pooled_output = outputs[0][:, 0]
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = None
    if labels is not None:
      loss_fct = torch.nn.MSELoss()
      loss = loss_fct(logits, labels.float())

    return {'loss': loss, 'logits': logits}

app = Flask(__name__)

# load tokenizer
print(f"loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# load numeric attribute model
print(f"loading sequence classification model...")
numeric_model = DistilBertForMultiLabelSequenceClassification.from_pretrained('./distilbertmodels/numeric_model')
numeric_model.eval()
# load string token classification model
print(f"loading token classification model...")
str_model = DistilBertForTokenClassification.from_pretrained('./distilbertmodels/string_model')
str_model.eval()

# define attribute names and corresponding index ranges
numeric_attributes = ["sentiment", "quality", "userExperience", "usability", "design", "durability", "pricing", "asAdvertised", "customerSupport", "repurchaseIntent"]
# define the mapping from ids to labels and sentiments
id2label_sentiment = {0: ('O', None), 1: ('B-NEG', 0), 2: ('I-NEG', 0), 3: ('B-NEU', 1), 4: ('I-NEU', 1), 5: ('B-POS', 2), 6: ('I-POS', 2)}

stop_words = ["a", "again", "all", "an", "and", "any", "are", "at", "both", "but", "by", "can", "could", "each", "false", "few", "for", "from", "further", "he", "her", "her", "here", "him", "his", "how", "i", "in", "is", "it", "more", "most", "no", "nor", "nor", "not", "of", "on", "once", "or", "other", "out", "over", "really", "seem", "she", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "they", "this", "to", "too", "true", "under", "us", "very", "we", "what", "when", "where", "why", "will", "would"]
stop_words_omitstart = ["again", "all", "and", "at", "both", "but", "by", "can", "each", "for", "from", "further", "here", "in", "of", "on", "or", "out", "over", "than", "that", "then", "to", "under", "when"]
stop_words_omitend = ["a", "all", "an", "and", "any", "are", "as", "both", "by", "from", "he", "her", "here", "his", "i", "in", "is", "it", "my", "of", "or", "other", "our", "same", "she", "some", "such", "than", "that", "the", "their", "there", "they", "we", "what", "when"]
stop_words_set = set(stop_words)
symbols_set = set([',', '.', '!', '?', ':', ';', '-'])

def extract_notable_attributes(preds, tokens):
  notable_attributes = []
  current_string = ""
  current_sentiment = None
  previous_label = None
  print(f"tokens: {tokens}")
  print(f"preds: {preds}")

  for token, label_id in zip(tokens, preds):
    label, sentiment = id2label_sentiment[label_id]
    token_text = token[2:] if token.startswith('##') else token

    if token == '[SEP]' or token == '[PAD]':
      if current_string and current_sentiment is not None:
        # flush previous
        notable_attributes.append([current_string.strip(), current_sentiment])
      break

    if (token_text in symbols_set or token.startswith('##')) and previous_label != 'O':
      # concatenate symbols or subwords if not first in sequence
      current_string += token_text
      continue

    if (label.startswith('B-') or label.startswith('I-')) and previous_label == 'O':
      # prev = O and current = B/I
      if current_string and current_sentiment is not None:
        # flush previous
        notable_attributes.append([current_string.strip(), current_sentiment])
      current_string = " " + token_text 
      current_sentiment = sentiment
    elif (label.startswith('B-') or label.startswith('I-')) and previous_label != 'O':
      # prev = B/I and current = B/I
      current_string += " " + token_text
    else:
      # current label is O
      if current_string and current_sentiment is not None:
        # flush previous
        notable_attributes.append([current_string.strip(), current_sentiment])
        current_string = ""
        current_sentiment = None

    previous_label = label

  if current_string:
    notable_attributes.append([current_string.strip(), current_sentiment])

  # filter out entries that contain only stopwords/symbols/empty string
  notable_attributes = [
    [attr[0].strip(" ,.!?:;-"), attr[1]] for attr in notable_attributes 
    if not set(attr[0].lower().split()).issubset(stop_words_set | symbols_set) 
    and not all(is_number(word) for word in attr[0].split())
    and len(attr[0].strip()) > 1
    and attr[1] is not None
  ]
  
  # remove inapplicable stopwords at either end of each attribute
  notable_attributes = [
    [' '.join(attr[0].split()[1:]).strip() if attr[0].split() and attr[0].split()[0].lower() in stop_words_omitstart else attr[0], attr[1]] for attr in notable_attributes
  ]
  notable_attributes = [
    [' '.join(attr[0].split()[:-1]).strip() if attr[0].split() and attr[0].split()[-1].lower() in stop_words_omitend else attr[0], attr[1]] for attr in notable_attributes
  ]
  
  return notable_attributes


@app.route('/predict', methods=['POST'])
def predict():
  data = request.json 
  print(f"received data for inference: {data}")
  # data is an array of review entries [{reviewID, content}]
  if not isinstance(data, list):
    return jsonify({'error': 'data should be a list'}), 400
  print(f"received inference request containing {len(data)} entries")
  predictions = []
  start_time = time.time()
  contents = [item['content'] for item in data]
  review_ids = [item['reviewID'] for item in data]
  encoding = tokenizer(contents, return_tensors='pt', padding=True, truncation=True, max_length=256)

  # batch inference for numeric attributes
  with torch.no_grad():
    num_outputs = numeric_model(**{k: v.to(numeric_model.device) for k, v in encoding.items()})
    predicted_scores_batch = num_outputs['logits'].tolist()

  # batch inference for token classification
  with torch.no_grad():
    str_outputs = str_model(**{k: v.to(str_model.device) for k, v in encoding.items()})
    str_predictions_batch = torch.argmax(str_outputs.logits, dim=-1).tolist()

  # postprocessing per entry in batch
  for idx, review_id in enumerate(review_ids):
    predicted_scores = predicted_scores_batch[idx]
    str_predictions = str_predictions_batch[idx]

    # process numeric attribute scores
    review_labels = {"reviewID": review_id}
    for i, attr in enumerate(numeric_attributes):
      score = min(max(round(predicted_scores[i] * 5), 0 if attr != 'sentiment' else 1), 5)
      review_labels[attr] = score

    # process string attribute scores
    notable_attributes = extract_notable_attributes(str_predictions, encoding.tokens(idx))
    review_labels['notableAttributes'] = notable_attributes
    
    predictions.append(review_labels)
    
  print(f"{len(data)} entries processed in {(time.time()-start_time):.2f} seconds")
  return jsonify(predictions)


if __name__ == '__main__':
  #app.run(debug=True)
  app.run(host='0.0.0.0', port=5000, debug=False)