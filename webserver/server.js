const fs = require('fs');
const path = require('path');
const axios = require('axios');
const express = require('express');
const cors = require('cors');
const subdomain = require('express-subdomain');
const app = express();
const bodyParser = require('body-parser');
const uaParser = require('ua-parser-js');
const SQLite = require("better-sqlite3");
const sql = new SQLite("./databases/data.sqlite", { verbose: console.log });
const puppeteer = require('puppeteer');

//launch a single browser instance to minimize latency between inferences
const browser = await puppeteer.launch({args: ['--no-sandbox', '--disable-setuid-sandbox']});

//reviews database
const reviews = sql.prepare("SELECT count(*) FROM sqlite_master WHERE type='table' AND name = 'reviews';").get();
if (!reviews["count(*)"]) {
  sql.prepare("CREATE TABLE reviews (id INTEGER PRIMARY KEY, content TEXT, sentiment INTEGER, quality INTEGER, userExperience INTEGER, usability INTEGER, design INTEGER, durability INTEGER, pricing INTEGER, asAdvertised INTEGER, customerSupport INTEGER, repurchaseIntent INTEGER, notableAttributes TEXT, labeler TEXT, timestamp INTEGER);").run();
  sql.prepare("CREATE UNIQUE INDEX idx_reviews_id ON reviews (id);").run();
  sql.pragma("synchronous = 1");
  sql.pragma("journal_mode = wal");
}
const getReviewEntry = sql.prepare("SELECT * FROM reviews WHERE id = ?");
const setReviewEntry = sql.prepare("INSERT OR REPLACE INTO reviews (id, content, sentiment, quality, userExperience, usability, design, durability, pricing, asAdvertised, customerSupport, repurchaseIntent, notableAttributes, labeler, timestamp) VALUES (@id, @content, @sentiment, @quality, @userExperience, @usability, @design, @durability, @pricing, @asAdvertised, @customerSupport, @repurchaseIntent, @notableAttributes, @labeler, @timestamp);");

function selectRandomUnlabelled(){
  let selected = sql.prepare(`SELECT * FROM reviews WHERE sentiment IS NULL ORDER BY RANDOM() LIMIT 1`).get();
  console.log(`selected review entry #${selected.id}.`);
  return selected;
}

const port = 80;
app.use(bodyParser.json()); 
app.use(express.static('public'));
app.use(cors({
  origin: 'https://www.amazon.com',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
app.set('views', __dirname + 'views');
app.set('view engine', 'ejs');
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET,HEAD,OPTIONS,POST,PUT');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
  next();
});

app.get('/entry', (req, res) => {
  console.log(`<DEBUG> review entry ${req.query.index ? `#${req.query.index} ` : ''}requested`);
  if(req.query.index) {
    res.json(getReviewEntry.get(req.query.index));
  } else {
    res.json(selectRandomUnlabelled());
  }
});

app.post('/submit', (req, res) => {
  const submittedData = req.body;
  console.log(`received submission for entry #${submittedData.id}: ${JSON.stringify(submittedData)}`);
  let reviewEntry = getReviewEntry.get(submittedData.id);

  const { sentiment, quality, userExperience, usability, design, durability, pricing, asAdvertised, customerSupport, repurchaseIntent, notableAttributes } = submittedData;
  
  Object.assign(reviewEntry, {
    sentiment,
    quality,
    userExperience,
    usability,
    design,
    durability,
    pricing,
    asAdvertised,
    customerSupport,
    repurchaseIntent,
    notableAttributes: notableAttributes ? JSON.stringify(notableAttributes) : "",
    labeler: req.ip,
    timestamp: Date.now(),
  });

  setReviewEntry.run(reviewEntry);
  res.send(`submitted data for entry #${submittedData.id}.`);
});

app.post('/predict', async (req, res) => {
  const reviewJson = req.body;
  console.log(`review inference requested, request=${JSON.stringify(reviewJson)}`);
  try {
    const response = await axios.post('http://127.0.0.1:5000/predict', reviewJson);
    console.log(`inference completed, sending data.`);
    res.json(response.data);
  } catch (error) {
    console.error('error during prediction:', error.message);
    if (error.response) {
      console.error('error data:', error.response.data);
      console.error('error status:', error.response.status);
      console.error('error headers:', error.response.headers);
    } else if (error.request) {
      console.error('error request:', error.request);
    } else {
      console.error('error message:', error.message);
    }
    res.status(500).send('error processing predictions (exception caught during axios call)');
  }
});


app.listen(port, () => {
  console.log(`listening on port ${port}`);
});