# LSTM Stock Predictor

![deep-learning.jpg](Images/deep-learning.jpg)

Due to the volatility of cryptocurrency speculation, investors will often try to incorporate sentiment from social media and news articles to help guide their trading strategies. One such indicator is the [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) which attempts to use a variety of data sources to produce a daily FNG value for cryptocurrency. I was asked to help build and evaluate deep learning models using both the FNG values and simple closing prices to determine if the FNG indicator provides a better signal for cryptocurrencies than the normal closing price data.

In this assignment, I used deep learning recurrent neural networks to model bitcoin closing prices. One model used the FNG indicators to predict the closing price while the second model used a window of closing prices to predict the nth closing price.

I completed the following:

1. [Prepare the data for training and testing](#prepare-the-data-for-training-and-testing)
2. [Build and train custom LSTM RNNs](#build-and-train-custom-lstm-rnns)
3. [Evaluate the performance of each model](#evaluate-the-performance-of-each-model)

- - -

### Files

[Closing Prices Notebook](lstm_stock_predictor_closing.ipynb)

[FNG Notebook](lstm_stock_predictor_fng.ipynb)

- - -

### Prepare the data for training and testing

I used the starter code as a guide to create a Jupyter Notebook for each RNN. The starter code contains a function to create the window of time for the data in each dataset.

For the Fear and Greed model, I used the FNG values to try and predict the closing price. A function was provided in the notebook to help with this.

For the closing price model, I used previous closing prices to try and predict the next closing price. A function was provided in the notebook to help with this.

Each model will used 70% of the data for training and 30% of the data for testing.

I applied a MinMaxScaler to the X and y values to scale the data for the model.

Finally, I reshaped the X_train and X_test values to fit the model's requirement of samples, time steps, and features. (*example:* `X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))`)

### Build and train custom LSTM RNNs

In each Jupyter Notebook, I created the same custom LSTM RNN architecture. In one notebook, I fit the data using the FNG values. In the second notebook, I fit the data using only closing prices.

I used the same parameters and training steps for each model. This was necessary to compare each model accurately.

### Evaluate the performance of each model

Finally, using the testing data to evaluate each model and compare the performance.

I was able to answer the following:

> Which model has a lower loss?

>   The closing price model returned a lower loss than the FNG model. 

> Which model tracks the actual values better over time?

>   The closing price model tracked actual values directionaly better than the FNG model, which was realtively flat.

> Which window size works best for the model?

>   I maintained a window size of ten for consistency but a window size of 1 performed the best.

- - -

### Resources

[Keras Sequential Model Guide](https://keras.io/getting-started/sequential-model-guide/)

[Illustrated Guide to LSTMs](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

[Stanford's RNN Cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks)

- - -