# Trained Models and datasets
Trained models and the datasets used can be found here: https://1drv.ms/u/s!AsshhlQM3x93qw6Bn44LfmTBreFz?e=Mdnpx2
Simply extract the folders into the project directory

# Requirements
Python 3.9.5
VS Code - with the Jupyter extension installed (Using version v2021.8.1046824664) - alternatively any other `.ipynb` viewer (iPython notebook)
- This is sufficient to view the iPython sessions that have been recorded in the `Trained Models` directory

executing `setup.sh` - creates a python virtual environment and installs necessary packages to it
- after doing this you can run all code from the command line (this is NOT RECOMMENDED)
- Instead follow the steps bellow to setup VENV with VSCode then run the scripts in python interactive mode
- to install on windows or mac, simply skip the first 3 lines in the setup.sh script and install the python packages without a virtual environment.

if a GPU is present it must have at least 8GB of memory, otherwise in each of the Claim Detection files change
`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
to
`device = "cpu"`
Using the CPU will increase the memory requirement from 2GB to 10GB

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

# Viewing Training Output
There are 8 notebooks corresponding to the 8/9 tasks which were developed solely for this project to the 8/9 tasks which were developed solely for this project
Each can has the the trained model 

## Notebooks
The session output from each training task has been logged in a iPython notebook, these files are denoted `.ipynb` and can be found under each training directory.

## TensorBoard
The loss and evaluation metrics have also been logged for each training task, to open these:
1. press `Ctrl+Shift+P`
2. type `Python: launch TensorBoard`
3. select `use current directory`
4. TensorBoard will now launch, the exact task can be selected by checking the relevant task in the `runs` selection menu


## Reconfigure Mode

There are various different modes for each of the 3 ML files (\verb|Masked LM|, \verb|NSP| and \verb|Claim Detection|), each are described in the code and corresponds to a task referenced in our implementation.
All can be changed by substituting the appropriate string from the list of supported modes. 

## Changing pre-trained model path

If a new model is trained, the path to the model in the following step must be updated. For instance, if a new Masked LM model is trained for Twitter, the Twitter NSP task must be updated to load the appropriate model path. This can be done by simply changing the \verb|MASKED_LM_PATH| to the new model's path.