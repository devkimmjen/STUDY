# 01. Specialization Introduction, A Conversation with Andrew Ng
Welcome. This specialization will teach you how to take machine learnings that you may have trained and deploy them in TensorFlow. Maybe you've trained models in Jupyter Notebook or in your laptop, but how do you take that model and have it be running 24/7, have it serve actual user queries, and create value? This course will teach you how to do all that. Yes. So we'll be taking a look at running your models, for example, in the browser with JavaScript, even be able to run them on your phone. So we're just going to have a lot of fun looking at models, being able to take what you need to do to your model to be able to convert it to run on all these different form factors. For you to be good at machine learning, one of the key skills will be not just the modeling but also the deployment. One of the most exciting deployment scenarios is in JavaScript, so that you can have a neural network, train right there in your web browser, and create an inference right there in your web browser. Yes. So go check it out. We're going to be studying all of that in the next course. So please go on to the next video.
***
# Training and inference using Tensorflow.js in JavaScript
# 02. Course Introduction, A Conversation with Andrew Ng
You might be used to thinking of neural network training and inference as requiring beefy GPUs, running a big data center or maybe a big GPU at least in your desktop. 

But modern web browsers have come a long ways and modern computers are quite fast and web browsers are now full flesh runtime environments. 

One of the exciting things about JS is that it allows you to do neural network training and inference right there in your web browser. >> And the performance is amazingly good too. 

So even if you're training when the browser and your desktop has a GP on it with the webgl layer, you can get some really really nice performance. 

And we're going to be having some fun with some scenarios around that later in the course. >> So I think it's really cool that a user can upload a picture to a web browser or grab an image from a webcam. 

And then have a neural network do training or inference right in the web browser without needing to send that image up to the cloud to be processed by a server. >> Yeah, so you're going to save time on the round tripping or if in a scenario if you're doing it on mobile for example that you don't even need to be connected. >> Possess user privacy as well and for some applications that's important. 

And so in this course you learn how to do all this yourself. Please go on to the next video.

# 03. A Few Words From Laurence
Hi everybody, and welcome to this course on TensorFlow.js. 

In this course, you're going to take a look at training and inference using JavaScripts. 

This will allow you to take your knowledge of Machine Learning models and use them in the browser as well as on backend servers like Node.js. 

We're going to start in the browser, and this week you're going to build some basic models using JavaScript, and you'll execute them in a simple web page. 

Let's start with a quick review of the design and architecture of TensorFlow.js, and you can see that here. 

The goal is twofold. First, we want to make it easy for you to code against it with a friendly high-level API, but you can also go lower into the APIs and program against them directly too. 

It's designed to run in the browser as well as an Node.js server. 

Up to this point, you've been programming convolutional neural networks and deep neural networks using the Keras API in TensorFlow. 

The layers API and TensorFlow.js looks and feels a lot like Keras. 

So what you've been learning, you'll be able to use albeit with a slightly different syntax due to using JavaScript instead of Python. 

The low-level APIs are called the core APIs. 

They're designed to work with a TensorFlow saved model formats, which in TensorFlow 2.0 is designed to be a standard file format which can be used across the Python APIs, the JavaScript once, and even TensorFlow Lite for mobile and embedded devices. 

The Core API then works with the browser and can take advantage of WebGL for accelerated training and inference. 

Also on Node.js, you can build server-side or terminal applications using it. 

These can then take advantage of CPUs, GPUs, and TPUs depending on what's available to your machine. 

We're going to build the simplest possible neural network. 

One which matches two sets of numbers to see that the relationship between them is y equals two x minus 1. 

We'll do that in JavaScript.

# 04. Building the Model

I think that's probably a great place to start to see how to build for TensorFlow.js. 

So first things first, you're going to need a web page. 

Let's create the simplest possible one, and you can see that here. 

Pretty straightforward, right? The next thing you'll need to do is add a script tag below the head and above the body to load the TensorFlow.js file. 

Here's the codes to do that. As we mentioned, we're going to build a model that infers the relationship between two numbers where y equals 2x minus 1. 

So let's do that now in JavaScript. 

We're going to do all of this in a separate script block. 

So let's get started. Okay. So we can start the script block like this. Make sure this is above the body tag in your HTML page. 

This code will define the model for you. 

So let's look at it line by line. The first line defines the model to be a sequential. In the Keras Methodology, when you define a neural network, you do so as a sequence of layers. 

So we're saying our model is sequential. 

The simplest possible neural network is one layer with one neuron. So we're only adding one dense layer to our sequence. 

This dense layer has only one neuron in it, as you can see from the units equals one parameter. We then compiled a neural network with a loss function and an optimizer as before. 

The loss function is Mean Squared Error, which works really well in a linear relationship like this one. The SGD and the optimizer stands for stochastic gradient descent as before. Model.summary just outputs the summary of the model definition for us. You can see this in the console outputs, but the summary will look like this, showing that it is a super simple neural network. 

You might wonder why there are two parameters, and that's because each neuron is trying to find a weight and the bias, i.e, if y equals wx plus c, then w is one parameter and c is the other, thus there are two. 

Next up, before the closing script tag at this code, this is the data that you'll use to train the neural network. Now, this is likely a little different than what you might be used to using from Python. So let me call that out somewhat. 

First, you'll notice that we're defining it as a tensor 2D, whereas in Python we were able to use a NumPy array. 

We don't have NumPy in JavaScript, so we're going a little lower. As its name suggests, when using a Tensor 2D, you have a two dimensional array or two one-dimensional arrays. 

So in this case you'll see that my training values are in one array, and the second array is the shape of those training values. 

So I'm using a set of 6x values in a one-dimensional array, and thus the second parameter is 6, 1, and I'll do the same for y. So if you tweak this code to add or remove parameters, remember to also add the second array to match its size.

# 05. Training the Model
Training should be an asynchronous function because it will take an indeterminate time to complete. So your next piece of code will call an asynchronous function called doTraining, which we haven't written yet, but when it completes, it will actually do something. Let's unpack this because it is important. Do training is an asynchronous function that you're creating in a few minutes. As mentioned before, training can take an indeterminate amount of time, and we don't want to block the browser while this is going on. So it's better to do it as an asynchronous function that calls us back when it's done. You call it and parse it the model that you just created. Then when it calls back, the model is trained, and at that point we can call model.predict. We can use this, for example, to try to predict the value for 10. But note how we parse the data into the model. We again have to create a Tensor 2D with the first dimension being an array containing the value that we want to predict, in this case 10, and the second being the size of that array, which in this case is one by one. The tensor will be fed into the model and the model will produce an output, which will then be displayed in the alert box. Given x equals 10, what do you think that output will be? You might think it's 19, but it's not, it's actually a value very close to 19, maybe 18.9899 or something like that. So now we've seen how to define our model, it's input data, and how to get the call back from an asynchronous function that trains the model, but we haven't trained it yet. So let's now look at the code for that. So here's the complete asynchronous function for training the model. This code should go at the top of the script block that you've been creating, so let's go through it line by line. As with Python model.fit is the function call to do the training. You're asking the neural network to fit the Xs to the Ys. As it's an asynchronous operation, in JavaScript, you await the result. You then parse it the Xs and the Ys as parameters. The rest of the parameters are adjacent lists, which you can see is enclosed in braces with each list item denoted by a name, followed by a colon, followed by a value. So for example, for 500 epochs, we enter it like this in the list. For callbacks instead of having a custom callback class, like we did in Python, we can just specify the callback in the list. The definition of the callback is itself a list with the list item on epoch end is defined as a function. This is a powerful aspect of JavaScript where I can add functions as list items. In this case, on epoch end gets the epoch number and the logs as parameters. So I can print out the epoch and the loss for that epoch. So that's it for building this model, and as you can see if you're familiar with TensorFlow, it's somewhat similar to what you do in Python and TensorFlow. All be it with some changes to accommodate JavaScript syntax. This will train our neural network to fit our Xs to our Ys to try to infer the rules between them. In this case, it's Y equals to X minus 1. In the next video, you'll see that in action, so please move to the next video.
# 06. First Example In Code
So here's the JavaScript that we've just been developing, and I'm using the brackets editor for this. Here you can see the asynchronous function. This asynchronous function's job is to do the training of the model, and then the training of the model is going to await the model.fit return. Model.fit just simply fits the Xs to the Ys, it does it for 500 epochs, and it specifies a set of callbacks. In this case on the EpochEnd, I just wanted to log the epoch and log the loss. When we build our model we build it as a tf.sequential,.sequential has a number of layers. In this case we have just one layer and that one layer is a dense with a single units and an input shape of one. We compile the model, and we compile it with a loss of meanSquaredError and an optimizer of stochastic gradient descent. We'll then output a model.summary and we can see that in the console later. We then specify our data using tensor.ds. These tensor2ds will have two dimensions, the first dimension is the data itself. As you can see here, it's six elements of data and a one-dimensional array. Our second elements of the tensor2d is the specification of that. We're telling it that we have a one-dimensional array with six elements. We do the same for the Ys. Then to do the actual training, we will call the do training function that we created earlier on. Because this function is asynchronous, because it's awaiting the model.fit, we'll say do training model when it's complete we'll have then clause. In the then clause will just alert the results of model.predict, model.predict if we want to predict the value of 10, we again pass in a tensor2d. That tensor2d will have the array with the value that we want to predict in this case 10, and the specification of that array, in this case it's a one by one array. That's it. That's everything that we'll do to actually train this model in the browser. Let's take a look at it. Here I have the Chrome browser open, and in the Chrome browser I'm going to train the model. I have the Developer Tools open in Chrome browser so that we can see the console log. So if I refresh it, we saw briefly the summary, and now we see the training feedback, our epochs and our loss. We're in later epochs now. It's going pretty fast and the loss is changing quite slowly. We can see it's 0.1365. So the loss is quite low. We can see the output that we were given from the alert dialog is the value of the model trying to infer the relationship for 10 which we would expect to be 19 but in this case it's 18.90. We can also see that it's outputting a tensor because we just alerted the results. As we can see here, we just alerted the results of model.predict. We could try to extract the value from the tensor because model.predict is going to return a tensor to us. But right now I just alerted the actual tensor itself, and we can see that's what the alert box gives us. It's a tensor containing an array and that array just has the value 18.90. So that's it for your first HTML page, and your first JavaScript training in the browser. If we want to take a quick look at the summary, we can just scroll up and we can see here was the summary because the summary was exported before the Loss started being printed as part of the training. We can see here on epoch zero our loss is really big, epoch one it was smaller, and our Loss got smaller, and smaller, and smaller as it converged. If we look closely at our loss, we began to hit diminishing returns like right around probably epoch 150 where we can see it's like 0.17, 0.16, 0.15. It ended up a 0.13 something around epoch 280, and it stayed in the 0.13 all the way to epoch 500. So we could probably reduce our training epochs a bit if we wanted to. Thank you and now it's time to move onto the next video.

***
# Quiz 1
1. What is the name of the API at the heart of TensorFlow.js which allows things like layers to be used?
- JS API
- TFJS API
- Core TF API
- Core API
2. How does TensorFlow.js use GPU acceleration in the browser?
- It works natively through GPU libraries in TensorFlow
- It doesn’t
- You access GPU through WebGL in the browser
- You have to install GPU runtimes for each browser, and explicitly use them

3. How can you use a TPU with TensorFlow.js?
- Only using Colab
- You can't
- You have to serve your JS from a GCP instance
- You can use Node.js on GCP and access TPU instances
4. Which of the following lines of code will correctly add a single dense layer containing a single neuron that takes a numeric input to a model using JavaScript?
- ```javascript
    model.add(tf.layers.dense({units: 1, inputShape:= [1]}));
    ```
- ```javascript
    model.add(tf.layers.dense({units= 1, inputShape: [1]}));
    ```
- ```javascript
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    ```
- ```javascript
    model.add(tf.layers({units: 1, inputShape: [1,1]}))
    ```
5. When creating data to input to a model using Python you could use a numpy array. How would you do it in JavaScript?
- Use a tensor2d containing the data
- Use a tensor2d contining a numpyjs array
- Use a numpyjs array
- Use a tensor2d containing the data and the shape of the data
6. If I train a model to detect a linear relationship (i.e. Y=2X-1), what line of code would output a prediction from that model for Y where X=10?
- ```javascript
    alert(model.predict(tf.tensor2d([10], [1,1])));
    ```
- ```javascript
    alert(model.predict([10], [1,1]));
    ```
- ```javascript
    alert(model.predict(10));
    ```
- ```javascript
    alert(model.predict(tf.tensor2d([10])));
    ```
7. When training a model, if I want to log training status at the end of an epoch, what is the name of the callback event you want to capture?
- OnEpochEnded
- OnEpochEnd
- EpochEnded
- EpochEnd

***
# Training Models with CSV Files
# 07. The Iris Dataset
So that was a pretty simple example and one way you had the numbers and arrays in memory. A more common scenario is whether data comes in from somewhere else, maybe a database connection or an imported dataset. One of the most common ways of getting data into an ML model and JavaScript ones are no exception, is in reading data from CSV files. TensorFlow.js gives you some tools that makes this much easier. In this video, we'll take a look at them and step through an example of how to read data from a CSV file, and use it to train a multi-class classifier. The data in question will be for the famous Iris database. If you haven't seen it or used it already, details about it are available at the UCI repository that you can see here. The dataset is a very simple one with 150 samples taken from three types of iris flower with 50 from each type. These samples have four features and a label. This plot by Nicoguaro shows them really nicely and shows the potential to build a classifier to do so.

### Iris Dataset Documentation
To know more about the Iris dataset please visit the [Iris dataset home page.](https://archive.ics.uci.edu/ml/datasets/iris)

# 08. Reading the Data

So let's dig in and take a look at how we could do this using JavaScript intensive load.js. Before we code let's take a quick look at our CSV file. One thing to note is that the first line should contain the column name definitions. Don't forget these or your code won't be able to take advantage of the shortcuts and tensorflow. You can see the first few lines here with features being measurements of sepal and petal length and the label being the species of flower that we're going to classify. Note that they're all in text and code will remix this into numbers so we can map the species to a neuron in the output layer. We'll start by putting an asynchronous function into a JavaScript block. It has to be asynchronous because we will be awaiting some values for example when we're training. To load the data from the CSV, you'll use code like this. It uses the tf.data.csv class to handle loading and parsing the data, but there are a number of things you'll need to pay attention to. First of all, the CSV is at a URL. I don't have the server or protocol details, which means it's going to try to load it from the same directory as the web page it's hosting it. But it's important to note that it isn't loading from the file system directly. It's going through the HTTP stack to get the file, so you'll need to run this code on a web server. I'm using the brackets IDE in this course and it has a built-in web server and I'd recommend you use that or something similar. tf.data.csv takes care of CSV management for you. It saves you writing a lot of code, so it's strongly recommended. To get the file all you need to do is make a call to it passing the URL. Tensorflow doesn't know from this file what your features are labels are so you have to at least flag, which column should be treated as label, you do that with this syntax. And note that species is hard coded and recognize. Look back to the CSV and you'll see that species is the column definition for the last column where we have the label. tf.data.csv is smart enough to use that string from the first line which is why you should leave the first line of column names in the CSV file. The data comes back from tf.data.csv as dictionaries, and for training we want to convert it into a raise. We also want to convert the strings defined in the labels into a one hot encoded array of label values. If you aren't familiar with one hot and coding yet, don't worry, I'll explain it in a moment. To do this we call the map method on the dictionary and tell it that we want sets of x's and y's back like this. The column that we specified as a label the species is in the set of y's and we can process that set to create an array of values. In this case creating an array of three values per label. The values in the label will be 2 0 and 1 with the position of the 1 being based on the species. So if it's setosa, it will be in the first element of the array. If it's virginica, it will be the second. And if it's versicolor, it will be in the third. This is the one hot encoding that I mentioned earlier. I'll explain why it's done that way in a moment. The values that weren't flagged as labels are in the x's set. So if I call objects.values(xs) on that, I will get back an array of arrays of their values. Each row in the data set had four features giving me a 4 by 1 array. And these are then loaded into an array with the length of the number of rows in the CSV in this case 150. So I will return that as my set of features that I'll train on. After processing the text labels into one hot encoded arrays, I can now call object.values on this to get the array of arrays back also. I now have data that can be lo

# 09. One-hot Encoding
Okay. So a few times I've mentioned one-hot encoding here and I'm sure you're wondering what that is and why I chose it. Well, here is the explanation. Imagine a neural network that performs three classifications using three output neurons like this. Maybe it's the rock paper and scissors where in this case, you would want your classification to have the first neuron close to one and the others close to zero representing a rock. Or in this case, you want the second neuron close to one and the others close to zero representing paper, and finally in this case for scissors, you would want the third neuron close to one and the others closer to zero. Thus, if we feed labels into the neural network when training it that represent the desired outputs, we would encode them in the representation that we would like to see in the outputs and that's one-hot encoding, i.e. one of the values in the array is the hot value, and in this case, we're doing exactly the same thing with the three species of iris. Let's look back at the code and we'll see that in action. We see that the const labels is an array with three elements. The first element is one if the species is setosa and zero otherwise. The second element is one if the species is virginica and zero otherwise, and finally, the third element is one if the species is versicolor, and zero otherwise.

***
# Quiz 2
1. Given a set of Xs and Ys, how would you one-hot encode a label based on the text or ‘rock’, ‘paper’, ‘scissors’ ?
- ```javascript
    trainingData.map(({xs, ys}) => {
    const labels = [ ys.label == ? 1 : 0, "rock", ys.label == ? 1 : 0, "paper",  ys.label == ? 1 : 0, "scissors"]
    }
  ```
- ```javascript
    trainingData.map(({xs, ys}) => {
    const labels = [ ys.label == ? 1 : 0, "rock", ys.label == ? 1 : 0, "paper",  ys.label == ? 1 : 0, "scissors"]
    }
  ```
- ```javascript
    trainingData.map(({xs, ys}) => {
    const labels = [ ys.label == ? 1 : 0, "rock", ys.label == ? 1 : 0, "paper",  ys.label == ? 1 : 0, "scissors"]
    }
  ```
- ```javascript
    trainingData.map(({xs, ys}) => {
    const labels = [ ys.label == ? 1 : 0, "rock", ys.label == ? 1 : 0, "paper",  ys.label == ? 1 : 0, "scissors"]
    }
  ```
***
# 10. Designing the NN
So we'll design a neural network that looks like this. At the top are the four features, in the middle is a hidden layer with five nodes, and at the bottom of the three nodes that we'll use for the classification. This is what it looks like in code. It's a familiar sequential model, but let's break that down line by line. First, we define the model as a tf.sequential. Then we add the hidden layer with five neurons. By specifying the input shape with the number of features, which is calculated to be four, we are effectively doing what [inaudible] did in Python earlier on. Then we add the three neurons at the bottom activating them with a Softmax function to get the probability that the pattern will match the neuron for that class of flower. Then we'll compile the model with the categorical cross-entropy loss and an AdamOptimizer with a learning rate of 0.06. To do the training, we use model.fitdataset. This is a version of fits that you haven't seen before, but it's nice in that you don't have to do a lot of data pre-processing. You've done it already by creating the data as a dataset. You can pass the data in as the first parameter as you can see here. Then you pass a list of JSON style name values with things like the epoch specified like this and the callbacks like this. The callbacks specifies the list itself and which we specify the behavior on epoch end. Well, we'll just log the epoch number and the current loss. So now if we want to use the model to do inference and get a prediction, we can create an input tensor with feature values and pass it to the predict method of the model. So here you declare it as a Tensor 2D with four values for the petal and sepal length and width, and then the shape of that Tensor. We then pass that to the predict method and we'll get a Tensor back with a prediction in it. So that's a look at the code for building a multi-class classifier trained by CSV data. In the next video, you'll see a screencast of this in action and you'll be able to try the code for yourself.

# 11. Iris Classifier In Code

Here's the source code that we've been looking at. Let's take a look at it piece by piece. First of all I'm defining an asynchronous function called run. This function will be run later on right here. Inside the asynchronous function, I'm going to do a number of things. First of all, I'm going to load the data and creating const csvUrl that's just pointing at the 'iris.csv' file. We can take a look at the 'iris.csv' file and it's as we expect, it's four values for the features, one value for the label and there's column names at the top in the first line. Then when I instantiate my tf.data.csv, the important thing is I'm giving it the URL of that csv and I'm specifying my columns. I'm also specifying that the Species column is a label. Again, if we look back at the data file, we'll see the most column is a species and that's the one that I'm setting up to be the label and this word species is what's actually driving it from here. Once this training data has been set up, now tensorflow.js will recognize that the Species column is a label and it will be part of the y set. Everything else will be the features and there will be in the x's set. We'll see that now. Next, I'm just going to set up a couple of variables. One is the number of features and the number of features is the column names.length minus one because I know I only have one label and then I'm hard-coding that I have a 150 samples in there. If we take a look, we'll see there's actually a 150 samples in it. There's a 151 rows, but the row number one is the titles. So now the data that's coming out of the tf.data.csv I need to convert into arrays and that's what this is going to do. So my converted data, I'm going to call the.map method on the training data which is this cons. This is going to have x's and y's. I'm going to keep the x's as they are and just convert them into an array with objects.values x's. So you can see this returns my x's are now my object dot values x's, but my y's I want to convert the text for setosa, virginica and versicolor into a hot encoded array where it's 100 for setosa 010 virginica and 001 for versicolor. So what I do is I create my labels array, it has three elements in it. You see the comma separated here and all I'm going to say is that the first element is one, if the species is setosa otherwise it's zero. The second element is one, if it's virginica otherwise it's zero. The third element is one, if it's versicolor otherwise it's zero. So then in my y's, I'm going to convert this array, the labels array and return that as my y's and then I'm just going to put these in batches of 10. Now, my converted data is going to be the data that I can use for training. We can see here in our model.fit data set it's converted data that gets parsed to it, but first we need to build a model. So I'm going to build the model by just setting it up as a tf.sequential and now I'm going to add a dense layer to that with five units in it and those five units are activated by sigmoid function. If you remember in Python, we had another layer on top of that which was flatten. In JavaScript we just say the input shape number of features. So that's my input shape which is going to be the four features that are being passed in, gives me the equivalent thing. So you see I don't have a flatten, I just go straight to my hidden layer dense, but I tell it it's in bad shape. My output layer is going to have three units in it because I have three classes. It's going to be activated by softmax because I want to try and emulate the hot encoding as much as possible where one of the neurons will light up for the actual class that's going to be and that's it. That's my DNN has just been created, very simple one as you can see. I'm going to compile the model and then the compiled model I'm going to make categorical cross entropy, my last function because remember we're trying to classify into three different categories and for my optimizer I'm just going to set the adam optimizer and I've set the training rates 0.06. You could tweak this thing and play with this to see if you can get better results. So now fitting is I'm going to await because it's an asynchronous function and then the await model.fit Data Sets converting it to data set converted data. I'm going to run it for a 100 epochs and then on every epoch end I want to do a call back where it's just going to console.log the epoch number and the current loss that has been measured. That's it for training the data. I have to await it here because if I do not, then this next line will execute immediately before this has finished training and you'll see really funky predictions because there will be operating on a model that isn't yet fully trained. But if I await it then you'll see that it will finish training before this line executes and this line I've just said and I'm parsing it a tensor value. This tensor is going to be four values for the four features and then the shape of the tensor one comma four. These four features I just pulled out of the file, so I know I'm going to be over fitting. So it's 44291402 44291402 I know is going to be a setosa, I'm just using that for testing it. Then I model.predict parsing it in that Tensor that I've just created and then this is going to parse me back a tensor and I'm just going to alert that tensor to see what we get. So let's see what this looks like when I'm running it. I switch into Chrome in my Developer Tools and refresh and here we go. Now we can see my loss is decreasing epoch by epoch. I set it to run for a 100 epochs. Once it reaches the a 100th epoch, then we'll see the alert dialog come up. There it is. We said we knew it was a setosa and it's coming up with a 0.99 percent chance that it was a setosa. So at least I can see from this one piece of test data that by model's executing correctly

# Week 1 Wrap up
This week you got an introduction to Machine Learning in the browser with TensorFlow.js. You saw how to build your first basic model -- an iris classifier -- by reading data from a CSV, one hot encoding the labels, and then training a model to recognize future data as a type of Iris plant. Next week you’ll take this further by using Convolutional Neural Networks in the browser that can recognize images of handwritten digits!

