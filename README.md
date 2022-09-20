# uniface project
GitHub Repository: https://github.com/DARAEDom/uniface

## About
The uniface project is my dissertation project that aimed to provide comparison of various ML/Deep learning models to compare their performance against ORL dataset. 

Modeles included in this project include:
- SVC
- CNN
- ResNet
- LSTM
- NN

## How to run the project
1. Make sure you have python version 3.8 or newer
2. Enable virtual environment if you need to
3. Install all dependencies through pip/pipx/conda using 
`python -m pip install -r requirements.txt` or `pip install -r requirements.txt`

## To run website
1. In commandline type `python uniface_main.py` or `py uniface_main.py`
2. Go to 127.0.0.1:5000, unless you change the default port
### Keep in mind to predict labels, you need to build the models first
### use Run models, they will generate needed files
### This is caused by the fact the pre-trained models exceed the allowed size

## Run models
1. Go to models/ directory
2  Find model you want to use
3. Type `python <model>` or `py <model>`
4. In the case of neural networks there is tensorboard utility available
5. Type `tensorboard --logdir logs/fit`
6, Follow the instruction, by default it starts web app at 127.0.0.1:6006
