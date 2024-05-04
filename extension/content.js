function sendMessageToBackground(action, data) {
  return new Promise((resolve, reject) => {
    browser.runtime.sendMessage({action, data}, response => {
      if (response && response.status === "success") {
        resolve(response.data);
      } else if (response && response.status === "error") {
        reject(response.error);
      } else {
        reject(new Error("no response from background script."));
      }
    });
  });
}

function restoreState() {
  sendMessageToBackground("getData", {key: "asin"}).then(currentASIN => {
    console.log(`fetched ASIN = "${currentASIN}", page ASIN = "${document.getElementById("asin").textContent.split(/ +/g).at(-1)}"`)
    if (currentASIN !== document.getElementById("asin").textContent.split(/ +/g).at(-1)) return false;
    console.log(`ASIN matches stored data, restoring to previous state.`);
    Promise.all([
      sendMessageToBackground("getData", {key: "predictions"}),
      sendMessageToBackground("getData", {key: "reviews"}),
      sendMessageToBackground("getData", {key: "filterBy"})
    ]).then(([predictions, reviews, filterBy]) => {
      if(filterBy){
        const filterOptions = document.querySelectorAll('.filter-option');
        const selectedClass = 'selected';
        filterOptions.forEach(option => {
          if(option.textContent === filterBy){
            option.classList.add(selectedClass);
          } else {
            option.classList.remove(selectedClass);
          }
        });
      }
      if (predictions && reviews) {
        displayResults(predictions, reviews, filterBy);
      }
    }).catch(error => console.error('error restoring state:', error));
  });
}


function extractReviews() {
  const reviews = [];
  const reviewElements = document.querySelectorAll('div[data-hook="review"]');

  reviewElements.forEach(el => {
    const reviewID = el.getAttribute('id');
    const verifiedPurchase = el.querySelector('span[data-hook="avp-badge"]') !== null;
    const hasImage = el.querySelector('img[alt="Customer image"]') !== null;
    const helpfulVotesElement = el.querySelector('span[data-hook="helpful-vote-statement"]');
    const helpfulVotes = helpfulVotesElement ? parseInt(helpfulVotesElement.textContent.replace(/\D/g, ''), 10) || 0 : 0;
    const contentElement = el.querySelector('span[data-hook="review-body"]');
    const content = contentElement ? contentElement.textContent.trim() : '';

    reviews.push({
      reviewID,
      verifiedPurchase,
      hasImage,
      helpfulVotes,
      content
    });
  });

  console.log(`extracted ${reviews.length} review entries from document.body.innerHTML.`);
  if(reviews.length) {for(let e of reviews) {console.log(JSON.stringify(e));}}

  return reviews;
}

function aggregatePredictions(combinedData, filterBy = 'all') {
  console.log(`aggregating preds with filter "${filterBy}", combinedData = ${JSON.stringify(combinedData)}`);
  let filteredPredictions = [];
  switch(filterBy) {
    case 'all':
      filteredPredictions = combinedData;
      break;
    case 'verified':
      filteredPredictions = combinedData.filter(entry => entry.verifiedPurchase);
      break;
    case 'helpful':
      filteredPredictions = combinedData.filter(entry => entry.hasImage || entry.helpfulVotes > 0);
      break;
    default:
      // handle nullish filter
      filteredPredictions = combinedData;
      break;
  }

  console.log(`proceeding with ${filteredPredictions.length} preds.`);
  document.getElementById('results-header').innerHTML = `<b>overall results</b><br>(${filteredPredictions.length} matching filter)`;
  if(!filteredPredictions.length) return [];
  
  const aggregatedPredictions = filteredPredictions.reduce((acc, entry) => {
    Object.keys(entry).forEach(key => {
      if (typeof entry[key] === 'number' && key !== 'reviewID') {
        // Initialize attribute if not already initialized
        if (!acc[key]) {
          acc[key] = {
            sum: 0,
            count: 0
          };
        }
        // include nonzero entries only
        if (entry[key] !== 0) {
          acc[key].sum += entry[key];
          acc[key].count += 1;
        }
      } else if (Array.isArray(entry[key]) && key === 'notableAttributes') {
        acc[key] = (acc[key] || []).concat(entry[key]);
      }
    });
    return acc;
  }, {});

  // calculate the average for numeric attributes
  Object.keys(aggregatedPredictions).forEach(key => {
    if (typeof aggregatedPredictions[key] === 'object' && aggregatedPredictions[key].count > 0) {
      aggregatedPredictions[key] = aggregatedPredictions[key].sum / aggregatedPredictions[key].count;
    }
  });

  // aggregate notable attributes, averaging sentiments
  if (aggregatedPredictions.notableAttributes) {
    const attributeMap = new Map();
    aggregatedPredictions.notableAttributes.forEach(item => {
      const [attr, sentiment] = item;
      attributeMap.set(attr, sentiment);
    });
    aggregatedPredictions.notableAttributes = Array.from(attributeMap, ([name, sentiment]) => [name, Math.round(sentiment)]);
  }
  return aggregatedPredictions;
}

function setRating(id, rating) {
  let elem = document.getElementById(id);
  let bar = elem.querySelector('div');
  let parent = elem.parentNode;

  let attributeSpan = parent.querySelector('.attribute-name');
  if (!attributeSpan) attributeSpan = elem.previousElementSibling || elem.nextElementSibling;

  let percentage = rating * 20;
  let initialWidth = parseFloat(bar.style.width) || 0;
  bar.style.width = `${initialWidth}%`;

  setTimeout(() => {
    bar.style.width = `${percentage}%`;
  }, 50);

  if (percentage <= 40) {
    bar.style.backgroundColor = `hsl(0, 100%, ${percentage + 20}%)`;
  } else if (percentage <= 70) {
    let lightness = 50 + (percentage - 40) / 3;
    bar.style.backgroundColor = `hsl(43, 90%, ${lightness}%)`;
  } else {
    let lightness = 60 - (percentage - 70) * 0.1;
    bar.style.backgroundColor = `hsl(120, 80%, ${lightness}%)`;
  }

  // adjust opacity if the rating is zero (undefined)
  if (rating === 0) {
    elem.style.opacity = '0.3';
    if (attributeSpan) {
      attributeSpan.style.opacity = '0.3';
    }
  } else {
    elem.style.opacity = '1';
    if (attributeSpan) {
      attributeSpan.style.opacity = '1';
    }
  }
}

function setFeatures(features) {
  const featuresText = document.getElementById('features-text');
  if (features?.length) {
    featuresText.textContent = '';
    featuresText.appendChild(document.createTextNode(`${features.length} thing${features.length > 1 ? 's' : ''} customers are saying about the product:`));
    featuresText.appendChild(document.createElement('br'));
  
    features.map(([attribute, sentiment], index) => {
      const span = document.createElement('span');
      span.textContent = index < features.length - 1 ? `"${attribute}"` : `"${attribute}"`;
      span.style.whiteSpace = 'nowrap';
  
      switch(sentiment){
        case 0:
          span.classList.add('sentiment-negative');
          break;
        case 1:
          span.classList.add('sentiment-neutral');
          break;
        case 2:
          span.classList.add('sentiment-positive');
          break;
      }
  
      featuresText.appendChild(span);
      if (span.offsetWidth > window.innerWidth) featuresText.insertBefore(document.createElement('br'), span);
      if (index < features.length - 1) {
        featuresText.appendChild(document.createTextNode(', '));
        const nextSpanWidth = document.createElement('span');
        nextSpanWidth.textContent = `"${features[index + 1][0]}"`;
        nextSpanWidth.style.visibility = 'hidden';
        nextSpanWidth.style.whiteSpace = 'nowrap';
        document.body.appendChild(nextSpanWidth);
        
        if (span.getBoundingClientRect().right + nextSpanWidth.offsetWidth > window.innerWidth) {
          featuresText.appendChild(document.createElement('br'));
        }
        document.body.removeChild(nextSpanWidth);
      }
    });
  } else {
    featuresText.textContent = '(no data)';
  }

  document.getElementById('features-div').style.display = 'block';
}

function setAllRatings(preds){
  console.log(`setting all rating fields. preds: ${JSON.stringify(preds)}`);
  setRating('rating-sentiment', preds.sentiment || 0);
  setRating('rating-quality', preds.quality || 0);
  setRating('rating-experience', preds.userExperience || 0);
  setRating('rating-usability', preds.usability || 0);
  setRating('rating-design', preds.design || 0);
  setRating('rating-durability', preds.durability || 0);
  setRating('rating-pricing', preds.pricing || 0);
  setRating('rating-asAdvertised', preds.asAdvertised || 0);
  setRating('rating-support', preds.customerSupport || 0);
  setRating('rating-repurchaseIntent', preds.repurchaseIntent || 0);
  setFeatures(preds.notableAttributes || []);
}

function displayResults(predictions, originalReviewEntries, filterBy = 'all'){
  if(!predictions) return console.error(`predictions is undefined on displayResults() call`);
  const combinedData = (predictions instanceof Array ? predictions : JSON.parse(predictions)).map(prediction => {
    // find the original review that matches the prediction reviewID
    const originalReviewEntry = originalReviewEntries.find(review => review.reviewID === prediction.reviewID);
    if(!originalReviewEntry) {
      console.log(`no originalReviewEntry found for review ${prediction.reviewID}, allocating defaults.`);
      return {...prediction, verifiedPurchase: false, hasImage: false, helpfulVotes: 0};
    }
    // combine the prediction and the original review properties
    return {
      ...prediction,
      verifiedPurchase: originalReviewEntry.verifiedPurchase,
      hasImage: originalReviewEntry.hasImage,
      helpfulVotes: originalReviewEntry.helpfulVotes
    };
  });

  // aggregate (average) prediction results according to the current filterBy ['all', 'verified', 'helpful']
  const aggregatedPreds = aggregatePredictions(combinedData, filterBy);

  // display aggregated results
  setAllRatings(aggregatedPreds);
  
  // show/hide appropriate elements
  document.getElementById('start-button').style.display = 'none';
  document.getElementById('results-title').style.display = 'block';
  document.getElementById('filter-select-container').style.display = 'block';
  document.getElementById('ratings').style.display = 'block';
}

browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
  //check active tab
  if (tabs[0].url.includes('amazon.com')) {
    if(['dp', 'gp', 'product'].some(str => tabs[0].url.includes(`/${str}/`))){
      document.getElementById('title').style.display = 'block';
      document.getElementById("asin").textContent = `product ID: ${(tabs[0].url.match(/\/(dp|gp)\/(\w+)(\/|\?|$)/))[2]}`;
      document.getElementById('asin').style.display = 'block';
      
      if(!restoreState()) document.getElementById('start-button').style.display = 'block';
    } else {
      document.getElementById('default-message').innerHTML = 'please navigate to a product page<br>to start attribute extraction.';
      document.getElementById('default-message').style.display = 'block';
    }

  } else {
    document.getElementById('default-message').innerHTML = 'please navigate to <a href="https://www.amazon.com/">Amazon</a><br>to get started.';
    document.getElementById('default-message').style.display = 'block';
  }
});


document.getElementById("start-button").addEventListener("click", async () => {
  document.getElementById('start-button').style.display = 'none';
  document.querySelector('.loader').style.display = 'block';

  sendMessageToBackground("fetchPredsByASIN", {asin: document.getElementById("asin").textContent.split(/ +/g).at(-1)})
    .then(async () => {
      Promise.all([
        sendMessageToBackground("getData", {key: "predictions"}),
        sendMessageToBackground("getData", {key: "reviews"}),
        sendMessageToBackground("getData", {key: "filterBy"})
      ]).then(([predictions, reviews, filterBy]) => {
        if (predictions && reviews) {
          document.querySelector('.loader').style.display = 'none';
          displayResults(predictions, reviews, filterBy);
        }
      }).catch(error => console.error('error fetching preds:', error));
    })
    .catch(error => {
      console.error('error fetching predictions:', error);
    });
});

const filterOptions = document.querySelectorAll('.filter-option');
const selectedClass = 'selected';
filterOptions.forEach(option => {
  option.addEventListener('click', async () => {
    document.querySelector(`.${selectedClass}`).classList.remove(selectedClass);
    option.classList.add(selectedClass);
    console.log(`filter switched to '${option.textContent}'`);
    sendMessageToBackground("setData", {filterBy: option.textContent});
    Promise.all([
      sendMessageToBackground("getData", {key: "predictions"}),
      sendMessageToBackground("getData", {key: "reviews"}),
      sendMessageToBackground("getData", {key: "filterBy"})
    ]).then(([predictions, reviews, filterBy]) => {
      if (predictions && reviews) {
        displayResults(predictions, reviews, filterBy || 'all');
      }
    }).catch(error => console.error('error switching filters:', error));
  });
});
