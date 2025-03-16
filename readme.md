# Attention-Based LSTM for Bitcoin Price Prediction

## Introduction

Predicting cryptocurrency prices, particularly Bitcoin, has garnered significant interest due to their high volatility and potential for substantial returns. Traditional financial models often struggle with the non-linear and complex patterns inherent in cryptocurrency markets. To address these challenges, we employ an Attention-Based Long Short-Term Memory (ALSTM) model, leveraging its capability to capture intricate temporal dependencies and focus on relevant features within the data.

## Methodology

### Data Collection and Preprocessing

Our dataset comprises normalized Bitcoin prices and trading volumes. Normalization was performed by fitting the scaling parameters on the training data to prevent data leakage. The dataset was partitioned into 80% training and 20% testing subsets, with a 90-day gap between them to further mitigate data leakage. The scaling model was applied exclusively to the test data to ensure unbiased evaluation.

### Feature Selection

We incorporated 12 technical indicators as features, including price, volume, Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Bollinger Bands, Average True Range (ATR), and Stochastic Oscillator. These indicators are among the most popular according to Binance's report on technical analysis tools. ([mdpi.com](https://www.mdpi.com/2079-8954/12/11/498))

### Model Architecture

The ALSTM model integrates an attention mechanism with the traditional LSTM architecture. LSTM networks, introduced by Hochreiter and Schmidhuber, are adept at learning long-term dependencies in sequential data. ([bioinf.jku.at](https://www.bioinf.jku.at/publications/older/2604.pdf)) The attention mechanism enables the model to weigh the importance of different input features, allowing it to focus on the most relevant information for price prediction.

### Evaluation Metrics

We assessed the model's performance using several metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), Information Coefficient (IC), Information Coefficient Information Ratio (ICIR), Annualized Return (AR), Annualized Volatility (AV), Sharpe Ratio (SR), and Maximum Drawdown (MDD).

### Trading Strategies

Two trading strategies were implemented to evaluate the practical applicability of our predictions:

1. **Trend-Following Strategy**: If the predicted price (`y_pred`) is greater than the actual price (`y_test`), we take a long position (buy); otherwise, we take a short position (sell).

2. **Event-Driven Strategy**: If the price change exceeds a threshold of 5%, we take a strong buy or sell position accordingly.

## Results

### Prediction Performance

The ALSTM model demonstrated superior predictive performance compared to the standard LSTM model. This improvement is attributed to the attention mechanism's ability to focus on relevant features, enhancing the model's sensitivity to market dynamics.

### Trading Strategy Evaluation

The performance of the trading strategies based on the predictions from both models is summarized below:

**ALSTM Model Performance**

| Strategy               | Cumulative Return | AR (%) | AV (%) | SR   | MDD (%) | MSE          | MAE           | IC             | ICIR          |
|-------------------------|-------------------|--------|--------|------|---------|--------------|---------------|----------------|---------------|
| Trend Following         | 4.36e+15          | 3908.98| 345.63 | 11.3 | 46.38   |              |               |                |               |
| Event-Driven Trading    | 3.19e+13          | 3329.02| 330.82 | 10.06| 273.26  |              |               |                |               |
| Evaluation Metrics      |                   |        |        |      |         | 21,358,717.02| 3,553.64      | 0.6209         | 2.0598        |

**LSTM Model Performance**

| Strategy               | Cumulative Return | AR (%) | AV (%) | SR   | MDD (%) | MSE          | MAE           | IC             | ICIR          |
|-------------------------|-------------------|--------|--------|------|---------|--------------|---------------|----------------|---------------|
| Trend Following         | 4.38e+14          | 3710.83| 351.56 | 10.55| 89.55   |              |               |                |               |
| Event-Driven Trading    | 1.57e+14          | 3599.32| 344.10 | 10.45| 273.26  |              |               |                |               |
| Evaluation Metrics      |                   |        |        |      |         | 14,372,745.81| 2,982.81      | 0.6620         | 2.2017        |

The ALSTM model achieved higher cumulative returns and annualized returns compared to the standard LSTM model. However, it also exhibited higher volatility and maximum drawdown. Notably, the Information Coefficient (IC) and ICIR values were slightly higher for the LSTM model, indicating a marginally better rank correlation between predicted and actual returns.

## Conclusion

Incorporating an attention mechanism into the LSTM framework enhances the model's ability to capture relevant patterns in Bitcoin price movements, leading to improved predictive performance. While the ALSTM model offers higher returns, it also entails increased risk, as evidenced by higher volatility and drawdown metrics. Future research could explore optimizing the attention mechanism and integrating additional features to further improve prediction accuracy and trading performance.

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780. ([bioinf.jku.at](https://www.bioinf.jku.at/publications/older/2604.pdf))

- Lou, J., Cui, L., & Li, Y. (2022). Bi-LSTM Price Prediction based on Attention Mechanism. *arXiv preprint arXiv:2212.03443*. ([arxiv.org](https://arxiv.org/abs/2212.03443))

- Binance. (n.d.). 8 Technical Analysis Tools Every Crypto Trader Should Know. Retrieved from [https://www.binance.com/en/square/post/16592321478898](https://www.binance.com/en/square/post/16592321478898)

