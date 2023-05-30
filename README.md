# TOWARDS PRACTICAL AUTOMATIC PIANO REDUCTION USING BERT WITH SEMI-SUPERVISED LEARNING

## Environment setup
Please first create an environment using conda or venv with python3.7. Then install the dependencies as in `requirements.txt`.
 
## The baseline `DBM` method
To run the `DBM` method, 
1. Prepare midi pieces to be used for building the database in the `data` folder.
2. Prepare also the piece to be reduced.
3. go to `DBM/build_up.ipynb` and modify the directories, then run the code!

## Evaluation
All objective evaluation codes are included within the `eval` folder.
1. Make any necessary directory changes in `eval/eval.py` 
2. Run the code. You will get a pickle file which contains a directionary of the 