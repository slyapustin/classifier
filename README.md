Django based app to train and use Tensorflow classifier

# Installations

 - Create virtual env `python3 -m venv .env`
 - Activate virtual environment `source .env/bin/activate`
 - Install requrements: `pip install -r app/requirements.txt`
 - Run initial app migrations `python3 app/manage.py migrate`
 - Train models for the first time `python3 app/manage.py train`
 - Run Django server `python3 app/manage.py runserver`
 - Go to http://localhost:8000 and play with it


# Some helpful links
 - [Tensorflow Text Classification – Python Deep Learning](https://sourcedexter.com/tensorflow-text-classification-python/)
