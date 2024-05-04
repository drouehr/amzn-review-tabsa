//main popup
browser.browserAction.onClicked.addListener(() => {
  browser.windows.create({
    type: "popup",
    url: browser.runtime.getURL("gui.html")
  });
});

function isReviewInEnglish(reviewElement) {
  // check for link offering translation to english
  const translateLink = reviewElement.querySelector('a[data-hook="cr-translate-this-review-link"]');
  return translateLink === null;
}

function parseHelpfulVotes(text) {
  const lowerText = text.toLowerCase();
  if (lowerText.includes("one person")) return 1;
  const match = lowerText.match(/\d+/);
  return match ? parseInt(match[0], 10) : 0;
}

async function compileReviewEntries(reviews, doc){
  let reviewData = [];
  reviews.forEach(review => {
    if (isReviewInEnglish(review)) {
      const reviewID = review.getAttribute('id');
      const titleElement = review.querySelector('a[data-hook="review-title"]');
      let reviewTitle = titleElement ? titleElement.textContent.trim() : "";
      reviewTitle = reviewTitle.replace(/\d\.\d out of \d stars/g, '').replace(/\n+/g, ' ').trim();
      const timestampElement = review.querySelector('span[data-hook="review-date"]');
      const dateRegex = /Reviewed in .*? on (.*)/;
      const dateMatch = timestampElement.textContent.match(dateRegex);
      const verifiedPurchaseElement = review.querySelector('span[data-hook="avp-badge"], span[data-hook="avp-badge-linkless"]');
      const reviewBodyElement = review.querySelector('span[data-hook="review-body"]');
      const helpfulVotesText = review.querySelector('span[data-hook="helpful-vote-statement"]')?.textContent || "";
      let content = reviewBodyElement ? reviewBodyElement.textContent.trim() : "no content";
      content = content.replace(/The media could not be loaded\./g, '').replace(/\n+/g, ' ').replace(/\s+/g, ' ').trim();
    
      reviewData.push({
        reviewID,
        timestamp: dateMatch ? Date.parse(dateMatch[1]) : 0,
        verifiedPurchase: !!verifiedPurchaseElement,
        hasImage: doc.querySelector(`div[id="${reviewID}_imageSection_main"]`) !== null,
        helpfulVotes: parseHelpfulVotes(helpfulVotesText),
        content: `${reviewTitle} | ${content}`
      });
    }
  });
  console.log(`scraped review data: ${JSON.stringify(reviewData)}`);
  return reviewData;
}

// listen for messages from popup
browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("message received in background:", message);
  switch (message.action) {
    case "fetchPredsByASIN":{
      const asin = message.data.asin;
      //corsanywhere requires manual access requests at https://cors-anywhere.herokuapp.com/corsdemo
      const urlThroughProxy = `https://cors-anywhere.herokuapp.com/https://www.amazon.com/product-reviews/${asin}/ref=cm_cr_arp_d_viewopt_srt?sortBy=recent&pageNumber=1`;

      console.log(`<bg> fetching via proxy: "${urlThroughProxy}"`);
      fetch(urlThroughProxy)
        .then(response => response.text())
        .then(async html => {
          const parser = new DOMParser();
          const doc = parser.parseFromString(html, "text/html");

          const captchaForm = doc.querySelector('form[action*="/errors/validateCaptcha"]');
          const captchaPrompt = doc.querySelector('div.a-box-inner h4');
          if (captchaForm && captchaPrompt && captchaPrompt.textContent.includes("Enter the characters you see below")) {
            // they're onto us :(
            console.error('<bg> error fetching predictions: CAPTCHA page detected.');
            sendResponse({status: "error", error: `captcha page detected!`});
          }

          const reviews = doc.querySelectorAll('[data-hook="review"]');
          console.log(`<bg> found ${reviews.length} review elements.`);
          let reviewData = await compileReviewEntries(reviews, doc);

          if(reviewData.length){
            browser.storage.local.set({reviews: reviewData});
            console.log(`<bg> sending json data:\n${JSON.stringify(reviewData)}`);
            fetch('http://localhost:8080/predict', {
              method: 'POST',
              headers: {'Content-Type': 'application/json'},
              body: JSON.stringify(reviewData.map(({ reviewID, content }) => ({ reviewID, content })))
            })
            .then(response => response.json())
            .then(predictions => {
              console.log(`<bg> inferences:  ${JSON.stringify(predictions)}`);
              browser.storage.local.set({predictions: predictions})
                .then(() => {
                  browser.storage.local.set({asin: asin});
                  console.log(`<bg> set stored ASIN to "${asin}".`);
                  sendResponse({status: "success"});
                });
            })
            .catch(error => {
              console.error('<bg> error fetching predictions:', error);
              sendResponse({status: "error", error: error.toString()});
            });
          } else {
            console.log(`<bg> no reviews scraped, exiting.`);
            sendResponse({status: "success"});
          }
        });
      return true;
    }

    case "getData":
      if(message.data && message.data.key){
        browser.storage.local.get(message.data.key).then(result => {
          if (result.hasOwnProperty(message.data.key)) {
            console.log(`<bg> sending data "${message.data.key}":`, result);
            sendResponse({status: "success", data: result[message.data.key]});
          } else {
            console.log('<bg> key not found:', message.data.key);
            sendResponse({status: "success", data: null});
          }
        }).catch(error => {
          console.error('<bg> error retrieving data:', error);
          sendResponse({status: "error", data: null, error: error.toString()});
        });
        return true;
    }

    case "setData":
      browser.storage.local.set(message.data)
        .then(() => {
          sendResponse({status: "success"});
        })
        .catch(error => {
          console.error('<bg> error storing data:', error);
          sendResponse({status: "error", error: error});
        });
      return true;
  }
  return true;
});
