import os

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


# path = '/run/user/12096878/gvfs/sftp:host=juniper/export/scratch1/ajt/CME_playground/data/cme_dat/'
path = './data/cme_dat/'
earth_cme_path = path + 'cme_dat1/'
path_cme_soho = path + 'cme_dat2/'
path_helcats = path + 'cme_dat3/'
dbm_path = path + 'cme_dat4/'

