Description of datasets:
| Name        | Total Shape           | Train Shape  |Test Shape |Number of features | Number of categorical features | Short description |
| ------------- |:-------------:| -----:|  -----:|  -----:|  -----:|  -----:|
| [Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction/overview) | 150K | 120K | 30K | 22 | 22 | Click prediction |
| [Amazon.com - Employee Access Challenge](https://www.kaggle.com/c/amazon-employee-access-challenge/overview) | 32.7K | 26.2K |6.5K|9|9|Predict an employee's access needs, given his/her job role|
| [OneTwoTrip Contest](https://boosters.pro/championship/onetwotrip_challenge/overview) | 196K | 156.8K |39.2K| 40 | 29 |Ticket return prediction |
|[Porto Seguro’s Safe Driver Prediction](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction)| 595K | 476K | 119K | 57 | 50 | Predict the probability that a driver will initiate an auto insurance claim in the next year|


Results (ROC-AUC score):
| | Click           | Employee  |OneTwoTrip| Driver |
| ------------- |:-------------:| -----:|  -----:|   -----:|  
| James Stein | 0.7184 | 0.7924 |  0.6679 | 0.6327 |
| Label| 0.7421 | 0.8490 |0.7036| **0.6338** |
| Frequency| **0.7539** | **0.8513** |**0.7103**| 0.6320 |
| Target, smoothing=0| 0.7145 | 0.7844 |0.6683| 0.6328 |
| Target, smoothing=1| 0.7215 | 0.7782 |0.6683| 0.6328 |
| Target, smoothing=2| 0.7175 | 0.7846 |0.6682| 0.6328 |
| WoE| 0.7280 | 0.8064 |0.6879| 0.6323 |
