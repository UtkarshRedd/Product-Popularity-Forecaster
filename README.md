# Introduction 
This web-app was designed to calculate popularities of products from a particular catalog containing grocery products sold on an e-retail app. Sales revenue and quantity data of every product and category were aggreagated and used to calculate the popularity of the product. For a specific date, a seasonal ARIMAX model was then trained to forecast the sales revenue and quantity data for the next 7 days. This was done for products being sold in every major city to provide city-wise and city-wise-category-wise product popularities.  

A short demo of the app is given below: -

https://github.com/UtkarshRedd/Product-Popularity-Forecaster/assets/29978378/c92735ac-6222-488b-a4e2-cf82509141ed


Dependencies: -
- [Python](https://www.python.org/) (3.8 or later recommended)
- [Flask](https://flask.palletsprojects.com/en/2.1.x/) (2.0.3)
- [Flask-Cors](https://flask-cors.readthedocs.io/en/latest/) (3.0.10)
- [Jinja2](https://jinja.palletsprojects.com/) (3.0.3)
- [joblib](https://joblib.readthedocs.io/en/latest/) (1.1.0)
- [numpy](https://numpy.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/) (3.0.9)
- [pandas](https://pandas.pydata.org/) (1.5.0)
- [pmdarima](https://alkaline-ml.com/pmdarima/) (1.8.4)
- [python-dateutil](https://dateutil.readthedocs.io/en/stable/) (2.8.2)
- [requests](https://requests.readthedocs.io/en/latest/) (2.27.1)
- [scikit-learn](https://scikit-learn.org/stable/) (1.0.2)
- [scipy](https://www.scipy.org/) (1.7.3)
- [statsmodels](https://www.statsmodels.org/stable/index.html) (0.12.2)
- [urllib3](https://urllib3.readthedocs.io/en/latest/) (1.26.8)
- [zipp](https://pypi.org/project/zipp/) (3.7.0)
- [pyarrow](https://arrow.apache.org/docs/python/)
- [fastparquet](https://fastparquet.readthedocs.io/en/latest/)
- [pathlib](https://docs.python.org/3/library/pathlib.html)


Built With
Python - The programming language used
Flask - The web framework used
Jinja2 - Template engine for Python
pmdarima - Time Series Forecasting

Data is present in the datasets folder.

All machine learning utilities are present in the models folder. Trained models for each city is present in pickle files in the models.sarimax.trained_models folder. 

config.py contains all the relevant configurable variables. 

# Docker
1. Build the docker file by running the command - docker build -t <name> .
2. Run the docker file on port 5002 by running the command - docker run -p 5002:5002 <name>
