const express = require('express')
const app = express();
const port = 5000

const mongoose = require('mongoose')
mongoose.connect('', {
    useNewUrlParser: true, useUnifiedTopology: true, useCreateIndex: true, useFindAndModify: false
}).then(() => console.log('mongoDB Connected...'))

    .catch (err => console.log(Error));

app.get('/', (req, res) => res.send('Hello World!!'))

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
