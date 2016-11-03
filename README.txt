1. Drop this folder and all contents into some directory in which the user has read/write access.  If you're going to run this from the command line, this directory will need to be written to the PYTHONPATH environment variable associated with whichever python shell being run （should be python 2.7).  Because this code creates and subsequently persists many images/models for historical tracking, it is best to treat this as a project and not a library to be added to python's lib directory.

2. Requires the following libraries:
scipy
scikit-learn
matplotlib
pandas
numpy
xgboost
pyearth
jaydebeapi  - if connection doesn't work, make sure jdk is up-to-date (yum java-1.8 java-devel-1.8?)

Some of these require a fortran/c compiler.  Much of this can probably be done using either venv on EvanServer or conda.

3. make sure 'model_engine' and 'make_predictions' are converted to .py format

4. 'generate models' is for users to manually generate an updated model.  This will require training data, to be uploaded in $PROJECT/data/training.  IMPORTANT:  For training the ensemble, use data out of sample for what was used to train the other models.  There is a subfolder for this in $PROJECT/data/training/ensemble

5. job scheduler will execute the following code:

python make_predictions.py
vwload output/$YESTERDAY.csv [compass database predictions_table]
