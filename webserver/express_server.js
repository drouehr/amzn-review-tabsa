const axios = require('axios');
const express = require('express');
const app = express();
const bodyParser = require('body-parser');

const port = 8080;
app.use(bodyParser.json()); 
app.use(express.static('public'));
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

app.listen(port, 'localhost', () => {
  console.log(`listening on port ${port}`);
});