# FRE7773 - Final Project - Kaggle: Optiver Volatility Prediction

## Description

Volatility describes fluctuations in prices of securities in the financial markets. It is of great importance for both individual and institutional investors, for giving them guidance on different securities' risk characteristics, thus helping them make judicious and appropriate investing decisions based on their unique risk preferences. In our Machine Learning project, we built models to predict 112 stocks' realized volatility over a 10-minute time window across different market sectors.

## Getting Started

### Dependencies

* OS, Linux System

### Installing

* Whole project can be accessed at https://github.com/IreneLXR/FRE7773_Final_Project
* Raw data from kaggle can be downloaded at https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/data
* Download engineered data at https://drive.google.com/drive/folders/1BfBEnrOOaogj4QfyX-D4X_VtkLCrzHHN, put the data in to /data folder

### Executing Project

* Make sure all packages in requirements.txt are installed to the correct version
```
pip install -r requirements.txt
```
* Put the engineered and raw data under the ./data folder

* Run my_flow.py under the right directory to generate the metadata folder
```
python3 my_flow.py run
```
* After running my_flow.py, run app.py to generate a web application that allows users to input stock id and time id to make a prediction on volatility
```
python3 app.py run
```
* Run model_selection.ipynb to get the overview of the pipeline of the project

## Tracking Result
* Comet tracking result:
![IMG_7556](https://user-images.githubusercontent.com/46698580/207930098-a59fcb58-4553-413d-a5e2-7ab77446cd7e.jpg)


## Deployment
* The project was deployed on AWS, as shown below:
<img width="973" alt="p1" src="https://user-images.githubusercontent.com/46698580/206819611-9375f8c3-9685-4285-a6eb-79e427eca3c5.png">
<img width="618" alt="p2" src="https://user-images.githubusercontent.com/46698580/206819747-44996e60-7e1f-4399-88c7-ec646207d1f2.png">

## Authors

Helen Zhang [@HelenZhang00](https://github.com/HelenZhang00)  
Layla Li [@IreneLXR](https://github.com/IreneLXR)  
Yu Gu [@guyuuuuu](https://github.com/guyuuuuu)


