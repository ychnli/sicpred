######################################################################
# The process for preprocessing CESM2 large ensemble data is a little 
# different than for ERA5. Specifically, we need to download the data
# in batches from the cloud, and then regrid them to a lower resolution
# grid. 
######################################################################

from src import util_cesm
from src import config_cesm

def main():
    # check consistency between downloaded data
    n_members = util_cesm.find_downloaded_vars()

    # Normalize 
    vars_to_normalize = config_cesm.VAR_NAMES
    vars_to_normalize.remove("temp")
    util_cesm.normalize_data(overwrite=False, verbose=2, vars_to_normalize=vars_to_normalize)

    # Prepare data pairs 
    

    

if __name__ == "__main__":
    main()