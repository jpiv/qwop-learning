const path = require('path');
const express = require('express');
const app = express();

const PORT = 8080;

app.use(express.static(path.resolve()));


app.listen(PORT)