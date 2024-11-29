######################################################################
# The process for preprocessing CESM2 large ensemble data is a little 
# different than for ERA5. Specifically, we need to download the data
# in batches from the cloud, and then regrid them to a lower resolution
# grid. 
######################################################################

from src import util_cesm
from src import config_cesm

def main():
    # Normalize 
    vars_to_normalize = config_cesm.VAR_NAMES
    vars_to_normalize.remove("temp")

    for var_name in vars_to_normalize:
        if var_name in ["icefrac", "icethick"]:
            divide_by_stdev = False
        else: 
            divide_by_stdev = True
        util_cesm.normalize_data(var_name, overwrite=False, verbose=2, divide_by_stdev=divide_by_stdev)

    # Prepare data pairs 


    

if __name__ == "__main__":
    main()