# amzn-review-tabsa
aspect-based sentiment extraction 

this project is a proof-of-concept attribute extraction system wrapped in a browser extension that leverages natural language processing (NLP) techniques to perform targeted attribute-based sentiment analysis (TABSA) on Amazon product reviews. the system scrapes customer reviews of the queried product and predicts the overall sentiment and corresponding sentiment strength, as well as ratings for other key attributes, based on the reviews and aggregates the ratings based on a customizable filter.

## dataset

the project utilizes a small subset (6000 entries) from the [McAuley-Lab/Amazon-Reviews-2023](https://amazon-reviews-2023.github.io/) dataset, where each review is annotated with the following metrics:

- `quality`: overall product quality on a scale of 1 (very poor) to 5 (excellent), or 0 if it cannot be inferred from the text.
- `userExperience`: quality of overall user experience on a scale of 1 (very negative) to 5 (very positive), or 0 if not mentioned.
- `usability`: product ease of use/user-friendliness on a scale of 1 (confusion-inducing) to 5 (intuitive), or 0 if not mentioned.
- `design`: product design quality/visual appeal on a scale of 1 (very negative) to 5 (very positive), or 0 if not mentioned.
- `durability`: product durability on a scale of 1 (very flimsy) to 5 (extremely durable), or 0 if not mentioned.
- `pricing`: pricing perception on a scale of 1 (practically a scam) to 5 (well worth it), or 0 if not mentioned.
- `asAdvertised`: whether the product specifications match what was advertised/expected, on a scale of 1 (not at all what was advertised) to 5 (matches specs perfectly), or 0 if not mentioned.
- `customerSupport`: quality of pre-or post-sale customer support on a scale of 1 (very negative) to 5 (very positive), or 0 if not mentioned.
- `repurchaseIntent`: likelihood of the customer to repurchase or recommend the product, on a scale of 1 (would not recommend) to 5 (highly recommend), or 0 if not mentioned.
- `notableAttributes`: unique features or characteristics mentioned about the reviewed product (that aren't already covered by the metrics above), extracted as exact substrings from the review content, along with the sentiment associated with that attribute (0 for negative, 1 neutral, 2 positive) as a JSON array, i.e. `[["attribute1": int(0-2)], ["attribute2": int(0-2)]]`.

## architecture

the project consists of the following components:

1. **browser extension**: the browser extension is responsible for scraping Amazon product reviews and sending them to the Express.js server for processing. It also displays the results of the attribute extraction in the browser.

2. **express.js server**: the Express.js server receives the review data from the browser extension and forwards the prediction requests to the Flask server.

3. **flask server**: the Flask server hosts the fine-tuned DistilBERT models and handles the inference process. It receives the prediction requests from the Express.js server, processes them using the appropriate model, and returns the results.

4. **DistilBERT models**: two separate DistilBERT (https://arxiv.org/abs/1910.01108) models are used for attribute extraction:
   - `DistilBertForMultiLabelSequenceClassification`: a modified instance of DistilBertForSequenceClassification for the numerical attributes with an overridden multi-attribute classification layer with 10 labels.
   - `DistilBertForTokenClassification`: used for the extraction of the `notableAttributes` tokens.

## installation

1. clone the repository:

   ```bash
   git clone https://github.com/drouehr/amzn-review-tabsa.git
   cd amzn-review-tabsa
   ```

2. (if retraining on an updated dataset) install the required dependencies for the training environment:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install transformers==4.39.2
   ```

3. install the required dependencies for the inference environment:

   ```bash
   pip install torch torchvision torchaudio
   pip install transformers==4.39.2
   ```

4. install the browser extension as a temporary addon (instructions for Firefox):
   - go to `about:debugging#/runtime/this-firefox`
   - click `Load Temporary Add-on`
   - select any file in the extension directory.

5. start the express server:

   ```bash
   cd express-server
   npm install
   npm start
   ```

6. start the flask server:

   ```bash
   cd flask-server
   python app.py
   ```

## usage

1. navigate to an Amazon product page in the browser.
2. click on the browser extension icon to activate the attribute extraction process.
3. the extension scrapes the reviews via a CORS proxy, sends them to the server for processing, and displays the results in the GUI.

## limitations and future work

- this project is a proof-of-concept and is intended for demonstration and testing purposes. it may not handle high traffic volume or concurrent requests properly.