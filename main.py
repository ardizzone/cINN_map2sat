import configparser
import sys
import os
import warnings

usage = "Usage: main.py <train|test|both> <target directory>"

assert len(sys.argv) == 3, usage

mode = sys.argv[1]
output_dir = sys.argv[2]

conf_file = os.path.join(output_dir, 'conf.ini')

assert mode in ['train', 'test', 'both'], usage
assert os.path.isdir(output_dir), usage + f'\nNo such directory: "{output_dir}"'

# initialize the options with the defaults, overwrite the ones specified.
args = configparser.ConfigParser()
args.read(os.path.join(os.getcwd(), 'default.ini'))

if os.path.isfile(conf_file):
    args.read(conf_file)
else:
    warnings.warn(f'No config file found under "{conf_file}", using default')

# to ensure reproducibility in case the defaults changed,
# save the entire set of current options too
conf_full = os.path.join(sys.argv[2], 'config_full.ini')
with open(conf_full, 'w') as f:
    args.write(f)

args['checkpoints']['output_dir'] = output_dir

try:
    args['data']['data_root_folder'] = os.environ['DATASET_DIR']
except KeyError:
    raise ValueError("Please set the DATASET_DIR environment variable")

if mode == 'test' or mode == 'both':
    import evaluation
    evaluation.test(args)

if mode == 'train' or mode == 'both':
    import train
    train.train(args)
