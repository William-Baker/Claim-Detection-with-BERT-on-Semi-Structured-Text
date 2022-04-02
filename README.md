# Requirements
Python 3.9.5
VS Code - with the Jupyter extension installed (Using version v2021.8.1046824664) - alternatively any other `.ipynb` viewer (iPython notebook)
- This is sufficient to view the iPython sessions that have been recorded in the `Trained Models` directory
executing `setup.sh` - creates a python virtual environment and installs necessary packages to it
- after doing this you can run all code from the command line (this is NOT RECOMMENDED)
- Instead follow the steps bellow to setup VENV with VSCode then run the scripts in python interactive mode

## VENV
Be sure change the interpreter to the python executable in the venv folder before running iPython snippets
1. Open VS Code
2. `Ctrl+Shift+P`
3. Python: Select Interpreter
4. select to venv/bin/python


# Order Of Execution

1. `Twitter/1. Fetch Tweets.py`
2. `Twitter/2. Twitter Clean.py`
3. `Twitter/Wenija Ma et al/extract_data.py`
4. `Twitter/3. Label Tweets.py`
5. `Claim Detection/Masked LM.py` In both `UKP` and `Twitter` modes
6. `Claim Detection/NSP.py` In both `UKP` and `Twitter` modes (After updating the pre-trained model paths with those produces in step 5)
7. `Claim Detection/Claim Detection.py` In all modes (After updating the pre-trained model paths with those produces in step 6)

