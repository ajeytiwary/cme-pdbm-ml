import os
import pathlib
# Replace with the name of your Dash app
# This will end up being part of the URL of your deployed app,
# so it can't contain any spaces, capitalizations, or special characters
#
# This name MUST match the name that you specified in the
# Dash App Manager
DASH_APP_NAME = 'cme-pdbm-ml'

DASH_APP_PRIVACY = 'public'

# Dash On-Premise is configured with either "Path based routing"
# or "Domain based routing"
# Ask your server administrator which version was set up.
# If a separate subdomain was created,
# then set this to `False`. If it was not, set this to 'True'.
# Path based routing is the default option and most On-Premise
# users use this option.
PATH_BASED_ROUTING = True


PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data/cme_dat/").resolve()

# path = '/run/user/12096878/gvfs/sftp:host=juniper/export/scratch1/ajt/CME_playground/data/cme_dat/'
path = DATA_PATH
earth_cme_path = path + 'cme_dat1/'
path_cme_soho = path + 'cme_dat2/'
path_helcats = path + 'cme_dat3/'
dbm_path = path + 'cme_dat4/'

