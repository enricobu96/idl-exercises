# Get paths to the data and results directories.
import os 
src_path = os.path.dirname(os.path.realpath(__file__))
assignment2_path = os.path.dirname(src_path)
data_dir = os.path.join(assignment2_path,'data')
model_dir = os.path.join(assignment2_path,'model')
results_dir = os.path.join(assignment2_path,'results')
