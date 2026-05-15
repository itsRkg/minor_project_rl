import configparser
import os

# Use __file__ so this works regardless of the caller's working directory
# (e.g. when imported from a Jupyter notebook in a different folder).
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.conf')

config = configparser.ConfigParser()
config.read(_CONFIG_PATH)

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default]
