import os
from eureka.lib import optimizers
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import pickle
from eureka.lib import readECF
from eureka.lib import exomast
import numpy as np
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import warnings
warnings.filterwarnings('ignore')
import astropy.io.fits as pf
import eureka.lib.sort_nicely as sn 
import eureka.lib.smoothing as sm
import eureka.S3_data_reduction.QL_params as evp
from datetime import datetime

import re
import pandas as pd
import eureka.lib.plots
import eureka.S5_lightcurve_fitting.s5_fit as s5
import eureka.S6_planet_spectra.s6_spectra as s6
import matplotlib.pyplot as plt
from matplotlib.image import imread
# from pptx import Presentation
# from pptx.util import Inches

import glob
import plotly.graph_objects as go


def divide_white_Stage4(optimizer_inputs_file, loc_sci_list, filepath_best_inputs_list):

    optimizer_params = optimizers.read_inputs(optimizer_inputs_file)

    eventlabel = optimizer_params['eventlabel']
    planet_name = optimizer_params['planet_name']
    bright_star = optimizer_params['bright_star']
    enable_exotic_ld = optimizer_params['enable_exotic_ld']
    # loc_sci = optimizer_params['loc_sci']
    exomast_file = optimizer_params['exomast_file']
    exotic_ld_direc = optimizer_params['exotic_ld_direc']
    exotic_ld_file = optimizer_params['exotic_ld_file']
    ld_file = optimizer_params['ld_file']
    ld_file_white = optimizer_params['ld_file_white']
    use_generate_ld = optimizer_params['use_generate_ld']

    # Load catalog of planet data from pickle file
    with open(exomast_file, 'rb') as f:
        all_planet_data = pickle.load(f)

    # Query the locally saved catalog for an exoplanet's data
    if planet_name in all_planet_data:
        planet_data = all_planet_data[planet_name]
        print(planet_data)
    else:
        print(f"No offline data found for {planet_name}. Retrieving planetary data from exoMAST (ONLINE)")
        planet_name = eventlabel
        planet_data = exomast.get_target_data(planet_name)[0]
    # else:
    #     print(f"No data found for {planet_name}")
        
        
    def create_run_string(event_label, directory_path):
        # Get today's date in YYYY-MM-DD format
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Initialize the run number to 1
        run_number = 1

        # Check if the directory exists
        if os.path.exists(directory_path):
            # List all files in the directory and find the highest run number
            for file in os.listdir(directory_path):
                if file.startswith(f"JointSpec_{today_date}_{event_label}_run"):
                    try:
                        current_run_number = int(file.split('_')[-1][3:])
                        run_number = max(run_number, current_run_number + 1)
                    except ValueError:
                        continue

        # Create the formatted string
        return f"JointSpec_{today_date}_{eventlabel}_run{run_number}"
    

    ## DEFINE MAIN FUNCTION FROM HERE TO END OF CELL 

    num_transits = len(loc_sci_list)

    for i in range(num_transits):

        filepath_best_inputs = filepath_best_inputs_list[i]
        loc_sci = loc_sci_list[i]
        
        # Split the string into parts
        parts = loc_sci.split('/')
        # Keep everything after the 3rd "/"
        loc_sci_raw = '/'.join(parts[3:])

        # Revised regular expression to match both 'Visit123-456' and 'Visit123'
        match = re.search(r"Visit(\d+(?:-\d+)?)", loc_sci)

        # Check if a match was found
        if match:
            visit_nums = match.group(1)  # Extracts the matched group, which are the visit numbers
        # elif not match:
        #     match = re.search(r"Visit(\d+)", loc_sci)
        #     if match:
        #         visit_nums = match.group(1)  # Extracts the matched group, which are the visit numbers
        else:
            visit_nums = "No visit numbers found"

        with open(filepath_best_inputs + "best_inputs.pkl", "rb") as f:
            best = pickle.load(f)


        ## HST ONLY - Retrieve xwindow & ywindow ##

        # Initialize event object
        ev  = evp.event_init()

        # Object
        ev.obj_list = []
        ev.img_list = []
        # Retrieve all files from science directory
        for fname in os.listdir(loc_sci):
            if fname.endswith("ima.fits"):
                filedir     = loc_sci +'/'+ fname
                header      = pf.getheader(filedir)
                if header['OBSTYPE'] == 'SPECTROSCOPIC':
                    ev.obj_list.append(filedir)
                elif header['OBSTYPE'] == 'IMAGING':
                    ev.img_list.append(filedir)
        ev.obj_list = sn.sort_nicely(ev.obj_list) 
        ev.img_list = sn.sort_nicely(ev.img_list) 
        ev.n_files  = len(ev.obj_list)
        ev.n_img    = len(ev.img_list)

        # Determine image size, filter/grism, scan height
        hdulist         = pf.open(ev.obj_list[0])
        nx              = hdulist['SCI',1].header['NAXIS1']
        ny              = hdulist['SCI',1].header['NAXIS2']
        ev.grism        = hdulist[0].header['FILTER']
        ev.detector     = hdulist[0].header['DETECTOR']
        ev.flatoffset   = [-1*hdulist['SCI',1].header['LTV2'], -1*hdulist['SCI',1].header['LTV1']]
        # ev.n_reads      = hdulist['SCI',1].header['SAMPNUM']
        ev.eventdir     = hdulist[0].header['ROOTNAME'][:6]
        # scanheight      = hdulist[0].header['SCAN_LEN']/0.121   #Pixels
        # ev.spec_width   = np.round(scanheight/2./ev.n_reads+6).astype(int) # Commented Out - RA2023
        # ev.fitbghw      = np.round(scanheight/2./ev.n_reads+6).astype(int) # Commented Out - RA2023

        # Updates from 11-30-2023
        ev.scanrate     = hdulist[0].header['SCAN_RAT']
        ev.n_reads      = hdulist[0].header['NSAMP'] - 1
        ev.scanheight   = ev.scanrate/0.121*hdulist[0].header['EXPTIME']
        ev.spec_width   = np.round(ev.scanheight/2./(ev.n_reads-1)+6).astype(int)
        ev.fitbghw      = ev.spec_width

        # Determine extraction box location 
        data            = hdulist['SCI',1].data 
        smdata          = sm.smoothing(data, [5,5])
        ydiff           = np.diff(np.sum(smdata,axis=1))[20:-20]  
        xdiff           = np.diff(np.sum(smdata,axis=0))[20:-20]   

        if bright_star is False:
            ev.ywindow         = [np.argmax(ydiff), np.argmin(ydiff)+40]
        if bright_star is True:
            # Modified ev.window for bright targets
            ywindow_min     = max(1, np.argmax(ydiff) - 40)  # Make sure smallest value is not less than the 1st pixel
            ev.ywindow      = [ywindow_min, np.argmin(ydiff) + 85]

        if ev.grism == 'G141':
            ev.xwindow         = [np.argmax(xdiff), np.argmin(xdiff)+40]  # Added to ev object returned as output - RA2023
            # ev.xwindow         = [np.argmax(xdiff)-5, np.argmin(xdiff)+45]  # Added to ev object returned as output - RA2023
        else:
            # G102 grism doesn't have a sharp cutoff on the blue edge
            ev.xwindow         = [np.argmin(xdiff)-145, np.argmin(xdiff)+40]  # Added to ev object returned as output - RA2023
        hdulist.close()

        print(f"ev.xwindow = {ev.xwindow}")
        print(f"ev.ywindow = {ev.ywindow}")

        best['xwindow_LB'] = ev.xwindow[0]
        best['xwindow_UB'] = ev.xwindow[1]
        best['ywindow_LB'] = ev.ywindow[0]
        best['ywindow_UB'] = ev.ywindow[1]



        # Setup Stages 3, 4, 5

        ecf_path = '.'+os.sep

        ## Setup Meta ##
        # Load Eureka! control file and store values in Event object
        s3_ecffile = 'S3_' + eventlabel + '.ecf'
        s3_meta_multspec = readECF.MetaClass(ecf_path, s3_ecffile)

        s4_ecffile = 'S4_' + eventlabel + '.ecf'
        s4_meta_multspec = readECF.MetaClass(ecf_path, s4_ecffile)
        # s4_meta_multspec.nspecchan = 1

        s5_ecffile = 'S5_' + eventlabel + '.ecf'
        s5_meta_multspec = readECF.MetaClass(ecf_path, s5_ecffile)
        # s5_meta_multspec.inputdir = 'Stage4'
        # s5_meta_multspec.outputdir = 'Stage5'

        s6_ecffile = 'S6_' + eventlabel + '.ecf'
        s6_meta_multspec = readECF.MetaClass(ecf_path, s6_ecffile)


        s3_meta_multspec.inputdir = loc_sci
        s3_meta_multspec.inputdir_raw = loc_sci_raw

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        if best['xwindow_LB']:
            s3_meta_multspec.xwindow = [best['xwindow_LB'], best['xwindow_UB']]
        else:
            best['xwindow_LB'] = ev.xwindow[0]
            best['xwindow_UB'] = ev.xwindow[1]
            s3_meta_multspec.xwindow = [best['xwindow_LB'], best['xwindow_UB']]

        if best['ywindow_LB']:
            s3_meta_multspec.ywindow = [best['ywindow_LB'], best['ywindow_UB']]
        else:
            best['ywindow_LB'] = ev.ywindow[0]
            best['ywindow_UB'] = ev.ywindow[1]
            s3_meta_multspec.ywindow = [best['ywindow_LB'], best['ywindow_UB']]

        # Manual Clipping
        # s5_meta_multspec.manual_clip = manual_clip_lists

        # Fit Method
        s5_meta_multspec.fit_method = 'emcee'

        # Hide Plots
        s5_meta_multspec.hide_plots = False

        # Overwrite default meta inputs with optimized Stage 3 and Stage 4 inputs
        # Stage 3
        s3_meta_multspec.diffthresh = best['diffthresh']
        s3_meta_multspec.bg_hw = best['bg_hw']
        s3_meta_multspec.spec_hw = best['spec_hw']
        s3_meta_multspec.p3thresh = best['p3thresh']
        s3_meta_multspec.median_thresh = best['median_thresh']
        s3_meta_multspec.window_len = best['window_len']
        s3_meta_multspec.p5thresh = best['p5thresh']
        s3_meta_multspec.p7thresh = best['p7thresh']
        # Stage 4
        s4_meta_multspec.diffthresh = best['diffthresh']
        s4_meta_multspec.bg_hw = best['bg_hw']
        s4_meta_multspec.spec_hw = best['spec_hw']
        s4_meta_multspec.p3thresh = best['p3thresh']
        s4_meta_multspec.median_thresh = best['median_thresh']
        s4_meta_multspec.window_len = best['window_len']
        s4_meta_multspec.p5thresh = best['p5thresh']
        s4_meta_multspec.p7thresh = best['p7thresh']
        s4_meta_multspec.drift_range = best['drift_range']
        s4_meta_multspec.highpassWidth = best['highpassWidth']
        s4_meta_multspec.sigma = best['sigma']
        s4_meta_multspec.box_width = best['box_width']
        # Stage 5
        s5_meta_multspec.diffthresh = best['diffthresh']
        s5_meta_multspec.bg_hw = best['bg_hw']
        s5_meta_multspec.spec_hw = best['spec_hw']
        s5_meta_multspec.p3thresh = best['p3thresh']
        s5_meta_multspec.median_thresh = best['median_thresh']
        s5_meta_multspec.window_len = best['window_len']
        s5_meta_multspec.p5thresh = best['p5thresh']
        s5_meta_multspec.p7thresh = best['p7thresh']
        s5_meta_multspec.drift_range = best['drift_range']
        s5_meta_multspec.highpassWidth = best['highpassWidth']
        s5_meta_multspec.sigma = best['sigma']
        s5_meta_multspec.box_width = best['box_width']

        if enable_exotic_ld is True:

            # Retrieve Values for Exotic-ld
            s4_meta_multspec.teff = planet_data['Teff']
            s4_meta_multspec.logg = planet_data['stellar_gravity']
            s4_meta_multspec.metallicity = planet_data['Fe/H']

            s5_meta_multspec.teff = planet_data['Teff']
            s5_meta_multspec.logg = planet_data['stellar_gravity']
            s5_meta_multspec.metallicity = planet_data['Fe/H']

            # Turn on compute_ld
            s4_meta_multspec.compute_ld = True

            # Turn on compute white
            s4_meta_multspec.compute_white = True

            # Specify 1D or 3D Grid Model
            s4_meta_multspec.exotic_ld_grid = '3D'

            # Path for exotic-ld ancillary files 
            s4_meta_multspec.exotic_ld_direc = exotic_ld_direc

            # Path for exotic-ld throughput file 
            s4_meta_multspec.exotic_ld_file = exotic_ld_file

            # Turn on use_generate_ld and enter paths for ld files
            s5_meta_multspec.use_generate_ld = use_generate_ld

            # Path for ld file (white) 
            s5_meta_multspec.ld_file = ld_file

            # Path for ld file (white) 
            s5_meta_multspec.ld_file_white = ld_file_white


        directory = s4_meta_multspec.topdir
        # directory = last_outputdir_S4
        if not os.path.exists(directory):
            os.makedirs(directory)


        # # Create Run String
        if i == 0:
            run_string = create_run_string(eventlabel, s3_meta_multspec.topdir + "DataAnalysis/HST/" + eventlabel + "/JointSpec/")

        ## Run Stage 3

        # Regular expression pattern to find "VisitXX-XX"
        # pattern = r"Visit\d+-\d+"
        # pattern = r"Visit\d+"
        pattern = r"Visit\d+(-\d+)?"

        # Replacement string, concatenating "Visit" with the new visit_nums
        replacement = "Visit" + visit_nums

        # Replacing "VisitXX-XX" in the filepath with "Visit" + visit_nums
        updated_inputdir = re.sub(pattern, replacement, s3_meta_multspec.inputdir)
        s3_meta_multspec.inputdir = updated_inputdir
        updated_inputdir_raw = re.sub(pattern, replacement, s3_meta_multspec.inputdir_raw)
        s3_meta_multspec.inputdir_raw = updated_inputdir_raw
        # print(updated_inputdir)
        # print(updated_inputdir_raw)

        s3_meta_multspec.outputdir_raw = "DataAnalysis/HST/" + eventlabel + "/JointSpec_WhiteResiduals/" + run_string + "/S3/Visit" + visit_nums
        s3_meta_multspec.outputdir = s3_meta_multspec.topdir + s3_meta_multspec.outputdir_raw

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_multspec)
        last_outputdir_S3 = s3_meta.outputdir


        ## Run Stage 4 in loop for each spec chan
        s4_meta_multspec.inputdir = last_outputdir_S3

        outputdir = s4_meta_multspec.outputdir
        outputdir_raw = s4_meta_multspec.outputdir_raw

        # # Update outputdir to run-specific folder
        # # outputdir = outputdir + run_string + "/"
        # outputdir = s4_meta_multspec.topdir + "JointSpec/" + run_string + "/S4"

        # Define initial conditions
        wave_min_start = 1.12
        wave_max_start = 1.14
        increment = 0.02
        num_channels = 1

        for c in range(num_channels):
            # Update wave_min and wave_max
            s4_meta_multspec.wave_min = wave_min_start + (c * increment)
            s4_meta_multspec.wave_max = wave_max_start + (c * increment)

            # Split the string into parts
            parts = loc_sci.split('/')
            # Keep everything after the 3rd "/"
            visit_num = '/'.join(parts[-1:])

            # channel_dir = s4_meta_multspec.topdir + "/JointSpec/S4/chan" + str(c) + "/" + visit_num + "/"
            channel_dir = s4_meta_multspec.topdir + "/JointSpec_WhiteResiduals/" + run_string + "/S4/chan" + str(c) + "/" + visit_num + "/"
            # channel_dir = outputdir + "/chan" + str(c) + "/" + visit_num + "/"
            s4_meta_multspec.outputdir = channel_dir

            # channel_dir_raw = "/JointSpec/S4/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            channel_dir_raw = "/JointSpec_WhiteResiduals/" + run_string + "/S4/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            # channel_dir_raw = "/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            s4_meta_multspec.outputdir_raw = channel_dir_raw

            # Check if the directory exists, create it if it doesn't
            if not os.path.exists(channel_dir):
                os.makedirs(channel_dir)

            s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_multspec, s3_meta=s3_meta)

    run_dir = s4_meta_multspec.topdir + "/JointSpec_WhiteResiduals/" + run_string + "/"

    return eventlabel, run_dir


def divide_white_Stage5(eventlabel, run_dir, transits_to_mask):

    def count_matching_folders(start_dir, pattern):
        """
        Recursively count folders in start_dir matching the given pattern.

        Parameters:
        - start_dir (str): The directory to start the search from.
        - pattern (str): Regular expression pattern to match folder names.

        Returns:
        - int: The number of folders matching the pattern.
        """
        matching_count = 0

        # Compile the regular expression pattern for better performance
        compiled_pattern = re.compile(pattern)

        for root, dirs, _ in os.walk(start_dir):
            # Filter directories in 'dirs' that match the pattern
            matching_dirs = [d for d in dirs if compiled_pattern.match(d)]
            matching_count += len(matching_dirs)

        return matching_count

    # Define the pattern for matching folder names
    pattern_chan = r'^chan\d{1,2}$'
    # Count the matching folders
    matching_chan_count = count_matching_folders(run_dir+'S4/', pattern_chan)
    print(f"Number of matching channel folders: {matching_chan_count}")

    for c in range(matching_chan_count):
        search_directory = run_dir + 'S4/chan' + str(c)
        pattern_to_find = r'ap\d{2}_bg\d{2}'

        found_paths = []

        for root, dirs, files in os.walk(search_directory):
            for dir_name in dirs:
                if re.match(pattern_to_find, dir_name):
                    found_path = os.path.join(root, dir_name)
                    found_paths.append(found_path)

        # Dynamically remove the specified bad transits for the current channel
        # Reverse sorting to ensure deleting from the end doesn't affect indices of earlier items
        if len(transits_to_mask) > 0:
            for idx in sorted(transits_to_mask[c], reverse=True):
                if idx < len(found_paths):
                    del found_paths[idx]

        inputdir = found_paths[0]
        inputdirlist = found_paths[1:]


        # Define the pattern for matching folder names (VisitXX-XX)
        # pattern_visit = r'^Visit\d{2}-\d{2}$'
        # pattern_visit = r'^Visit\d{2}$'
        pattern_visit = r'^Visit\d{2}(-\d{2})?$'
        # Count the matching folders
        matching_visit_count = count_matching_folders(search_directory, pattern_visit)
        print(f"Number of matching visit folders: {matching_visit_count}")



        # Perform Joint Light Curve Fit

        eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

        ecf_path = '.'+os.sep
        s5_ecffile = 'S5_' + eventlabel + '.ecf'
        s5_meta_multspec = readECF.MetaClass(ecf_path, s5_ecffile)

        outputdir = run_dir + 'S5/chan' + str(c)

        parts = inputdir.split('/')
        # Keep everything after the 6th "/"
        inputdir_raw = '/'.join(parts[6:])

        parts = outputdir.split('/')
        # Keep everything after the 6th "/"
        outputdir_raw = '/'.join(parts[6:])

        s5_meta_multspec.multwhite = True
        s5_meta_multspec.inputdir = inputdir
        s5_meta_multspec.inputdirlist = inputdirlist
        s5_meta_multspec.inputdir_raw = inputdir_raw
        s5_meta_multspec.outputdir = outputdir 
        s5_meta_multspec.outputdir_raw = outputdir_raw

        # print(inputdir)
        # print(s5_meta_multspec.inputdir_raw)
        # print(inputdirlist)
        # print(outputdir)
        # print(s5_meta_multspec.outputdir_raw)

        # Set Fit Method
        s5_meta_multspec.fit_method = 'emcee'  

        s5_meta = s5.fitlc(eventlabel, ecf_path=ecf_path, s4_meta=None, input_meta=s5_meta_multspec) 
        # s5_meta = s5.fitlc(eventlabel, ecf_path=ecf_path, s4_meta=None)  # Shared

    def get_channel_number(dir_name):
        """
        Extracts the numerical part from a directory name formatted as 'chanX'.
        """
        try:
            return int(dir_name[4:])
        except ValueError:
            return -1  # Return -1 for any directory names that do not match expected format

    def process_images_in_figs_dir(figs_dir, chan_number):
        """
        Process all images in a given 'figs' directory within a channel folder and create a subplot compilation.
        """
        images = []
        image_files = [f for f in os.listdir(figs_dir) if f.endswith("_lc_emcee.png")]
        # Sorting the image files to ensure they are processed in order
        image_files.sort()
        for file in image_files:
            file_path = os.path.join(figs_dir, file)
            images.append(imread(file_path))
                
        if images:
            n = len(images)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            if n > 1:
                axs = axs.ravel()
            else:
                axs = [axs]
            for ax in axs[len(images):]:
                ax.axis('off')
            for img, ax in zip(images, axs):
                ax.imshow(img)
                ax.axis('off')
            plt.suptitle(f"Channel {chan_number}", fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_path = os.path.join(figs_dir, f"compilation_chan{chan_number}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Compilation saved: {output_path}")
        else:
            print(f"No images found in {figs_dir}")

    def find_and_process_figs_dir(run_dir):
        """
        Recursively search for 'chanX' directories to find 'figs' subdirectories and process the images within, ensuring channels are processed sequentially by channel number.
        """
        chan_dirs = []
        for root, dirs, files in os.walk(run_dir):
            for dir in dirs:
                if dir.startswith("chan"):
                    chan_dirs.append(os.path.join(root, dir))
        
        # Sort the channel directories by the numerical part of their names
        chan_dirs.sort(key=lambda x: get_channel_number(os.path.basename(x)))
        
        for chan_dir in chan_dirs:
            chan_number = get_channel_number(os.path.basename(chan_dir))
            if chan_number != -1:  # Process only if the folder name matches the expected format
                for sub_root, sub_dirs, sub_files in os.walk(chan_dir):
                    sub_dirs.sort()  # Sort sub_dirs to process them in order
                    if "figs" in sub_dirs:
                        figs_dir = os.path.join(sub_root, "figs")
                        process_images_in_figs_dir(figs_dir, str(chan_number))
                        break  # Assume only one 'figs' folder per channel folder

    # Example usage:
    find_and_process_figs_dir(run_dir)



def Stage4(optimizer_inputs_file, loc_sci_list, filepath_best_inputs_list):

    optimizer_params = optimizers.read_inputs(optimizer_inputs_file)

    eventlabel = optimizer_params['eventlabel']
    planet_name = optimizer_params['planet_name']
    bright_star = optimizer_params['bright_star']
    enable_exotic_ld = optimizer_params['enable_exotic_ld']
    # loc_sci = optimizer_params['loc_sci']
    exomast_file = optimizer_params['exomast_file']
    exotic_ld_direc = optimizer_params['exotic_ld_direc']
    exotic_ld_file = optimizer_params['exotic_ld_file']
    ld_file = optimizer_params['ld_file']
    ld_file_white = optimizer_params['ld_file_white']
    use_generate_ld = optimizer_params['use_generate_ld']

    # Load catalog of planet data from pickle file
    with open(exomast_file, 'rb') as f:
        all_planet_data = pickle.load(f)

    # Query the locally saved catalog for an exoplanet's data
    if planet_name in all_planet_data:
        planet_data = all_planet_data[planet_name]
        print(planet_data)
    else:
        print(f"No offline data found for {planet_name}. Retrieving planetary data from exoMAST (ONLINE)")
        planet_name = eventlabel
        planet_data = exomast.get_target_data(planet_name)[0]
    # else:
    #     print(f"No data found for {planet_name}")
        
        
    def create_run_string(event_label, directory_path):
        # Get today's date in YYYY-MM-DD format
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Initialize the run number to 1
        run_number = 1

        # Check if the directory exists
        if os.path.exists(directory_path):
            # List all files in the directory and find the highest run number
            for file in os.listdir(directory_path):
                if file.startswith(f"JointSpec_{today_date}_{event_label}_run"):
                    try:
                        current_run_number = int(file.split('_')[-1][3:])
                        run_number = max(run_number, current_run_number + 1)
                    except ValueError:
                        continue

        # Create the formatted string
        return f"JointSpec_{today_date}_{eventlabel}_run{run_number}"
    ## DEFINE MAIN FUNCTION FROM HERE TO END OF CELL 
    ## e.g. multspec_wfc3_Stage4(loc_sci_list, filepath_best_inputs_list)

    # filepath_best_inputs = "/Users/ashtar1/DataAnalysis/HST/HD86226c/Optimized/Visit04-07/Optimized_2024-03-08_HD86226c_run1/"
    # loc_sci = "/Users/ashtar1/Data/HST/WFC3/HD86226c/Visit04-07"
    ## visit = "04-07"  # e.g. "04-07"

    num_transits = len(loc_sci_list)

    for i in range(num_transits):

        filepath_best_inputs = filepath_best_inputs_list[i]
        loc_sci = loc_sci_list[i]
        
        # Split the string into parts
        parts = loc_sci.split('/')
        # Keep everything after the 3rd "/"
        loc_sci_raw = '/'.join(parts[3:])

        # Revised regular expression to match both 'Visit123-456' and 'Visit123'
        match = re.search(r"Visit(\d+(?:-\d+)?)", loc_sci)

        # if match:
        #     print("Match found:", match.group(1))
        # else:
        #     print("No match found.")


        # # Keep the visitID
        # # Regular expression to find the visit numbers after "Visit"
        # # match = re.search(r"Visit(\d+-\d+)", loc_sci)
        # match = re.search(r"Visit(\d+)", loc_sci)

        # Check if a match was found
        if match:
            visit_nums = match.group(1)  # Extracts the matched group, which are the visit numbers
        # elif not match:
        #     match = re.search(r"Visit(\d+)", loc_sci)
        #     if match:
        #         visit_nums = match.group(1)  # Extracts the matched group, which are the visit numbers
        else:
            visit_nums = "No visit numbers found"

        with open(filepath_best_inputs + "best_inputs.pkl", "rb") as f:
            best = pickle.load(f)




        ## HST ONLY - Retrieve xwindow & ywindow ##

        # Initialize event object
        ev  = evp.event_init()

        # Object
        ev.obj_list = []
        ev.img_list = []
        # Retrieve all files from science directory
        for fname in os.listdir(loc_sci):
            if fname.endswith("ima.fits"):
                filedir     = loc_sci +'/'+ fname
                header      = pf.getheader(filedir)
                if header['OBSTYPE'] == 'SPECTROSCOPIC':
                    ev.obj_list.append(filedir)
                elif header['OBSTYPE'] == 'IMAGING':
                    ev.img_list.append(filedir)
        ev.obj_list = sn.sort_nicely(ev.obj_list) 
        ev.img_list = sn.sort_nicely(ev.img_list) 
        ev.n_files  = len(ev.obj_list)
        ev.n_img    = len(ev.img_list)

        # Determine image size, filter/grism, scan height
        hdulist         = pf.open(ev.obj_list[0])
        nx              = hdulist['SCI',1].header['NAXIS1']
        ny              = hdulist['SCI',1].header['NAXIS2']
        ev.grism        = hdulist[0].header['FILTER']
        ev.detector     = hdulist[0].header['DETECTOR']
        ev.flatoffset   = [-1*hdulist['SCI',1].header['LTV2'], -1*hdulist['SCI',1].header['LTV1']]
        # ev.n_reads      = hdulist['SCI',1].header['SAMPNUM']
        ev.eventdir     = hdulist[0].header['ROOTNAME'][:6]
        # scanheight      = hdulist[0].header['SCAN_LEN']/0.121   #Pixels
        # ev.spec_width   = np.round(scanheight/2./ev.n_reads+6).astype(int) # Commented Out - RA2023
        # ev.fitbghw      = np.round(scanheight/2./ev.n_reads+6).astype(int) # Commented Out - RA2023

        # Updates from 11-30-2023
        ev.scanrate     = hdulist[0].header['SCAN_RAT']
        ev.n_reads      = hdulist[0].header['NSAMP'] - 1
        ev.scanheight   = ev.scanrate/0.121*hdulist[0].header['EXPTIME']
        ev.spec_width   = np.round(ev.scanheight/2./(ev.n_reads-1)+6).astype(int)
        ev.fitbghw      = ev.spec_width

        # Determine extraction box location 
        data            = hdulist['SCI',1].data 
        smdata          = sm.smoothing(data, [5,5])
        ydiff           = np.diff(np.sum(smdata,axis=1))[20:-20]  
        xdiff           = np.diff(np.sum(smdata,axis=0))[20:-20]   

        if bright_star is False:
            ev.ywindow         = [np.argmax(ydiff), np.argmin(ydiff)+40]
        if bright_star is True:
            # Modified ev.window for bright targets
            ywindow_min     = max(1, np.argmax(ydiff) - 40)  # Make sure smallest value is not less than the 1st pixel
            ev.ywindow      = [ywindow_min, np.argmin(ydiff) + 85]

        if ev.grism == 'G141':
            ev.xwindow         = [np.argmax(xdiff), np.argmin(xdiff)+40]  # Added to ev object returned as output - RA2023
            # ev.xwindow         = [np.argmax(xdiff)-5, np.argmin(xdiff)+45]  # Added to ev object returned as output - RA2023
        else:
            # G102 grism doesn't have a sharp cutoff on the blue edge
            ev.xwindow         = [np.argmin(xdiff)-145, np.argmin(xdiff)+40]  # Added to ev object returned as output - RA2023
        hdulist.close()

        print(f"ev.xwindow = {ev.xwindow}")
        print(f"ev.ywindow = {ev.ywindow}")

        best['xwindow_LB'] = ev.xwindow[0]
        best['xwindow_UB'] = ev.xwindow[1]
        best['ywindow_LB'] = ev.ywindow[0]
        best['ywindow_UB'] = ev.ywindow[1]




        # Setup Stages 3, 4, 5

        ecf_path = '.'+os.sep

        ## Setup Meta ##
        # Load Eureka! control file and store values in Event object
        s3_ecffile = 'S3_' + eventlabel + '.ecf'
        s3_meta_multspec = readECF.MetaClass(ecf_path, s3_ecffile)

        s4_ecffile = 'S4_' + eventlabel + '.ecf'
        s4_meta_multspec = readECF.MetaClass(ecf_path, s4_ecffile)

        s5_ecffile = 'S5_' + eventlabel + '.ecf'
        s5_meta_multspec = readECF.MetaClass(ecf_path, s5_ecffile)

        s6_ecffile = 'S6_' + eventlabel + '.ecf'
        s6_meta_multspec = readECF.MetaClass(ecf_path, s6_ecffile)


        s3_meta_multspec.inputdir = loc_sci
        s3_meta_multspec.inputdir_raw = loc_sci_raw

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        if best['xwindow_LB']:
            s3_meta_multspec.xwindow = [best['xwindow_LB'], best['xwindow_UB']]
        else:
            best['xwindow_LB'] = ev.xwindow[0]
            best['xwindow_UB'] = ev.xwindow[1]
            s3_meta_multspec.xwindow = [best['xwindow_LB'], best['xwindow_UB']]

        if best['ywindow_LB']:
            s3_meta_multspec.ywindow = [best['ywindow_LB'], best['ywindow_UB']]
        else:
            best['ywindow_LB'] = ev.ywindow[0]
            best['ywindow_UB'] = ev.ywindow[1]
            s3_meta_multspec.ywindow = [best['ywindow_LB'], best['ywindow_UB']]

        # Manual Clipping
        # s5_meta_multspec.manual_clip = manual_clip_lists

        # Fit Method
        s5_meta_multspec.fit_method = 'emcee'

        # Hide Plots
        s5_meta_multspec.hide_plots = False

        # Overwrite default meta inputs with optimized Stage 3 and Stage 4 inputs
        # Stage 3
        s3_meta_multspec.diffthresh = best['diffthresh']
        s3_meta_multspec.bg_hw = best['bg_hw']
        s3_meta_multspec.spec_hw = best['spec_hw']
        s3_meta_multspec.p3thresh = best['p3thresh']
        s3_meta_multspec.median_thresh = best['median_thresh']
        s3_meta_multspec.window_len = best['window_len']
        s3_meta_multspec.p5thresh = best['p5thresh']
        s3_meta_multspec.p7thresh = best['p7thresh']
        # Stage 4
        s4_meta_multspec.diffthresh = best['diffthresh']
        s4_meta_multspec.bg_hw = best['bg_hw']
        s4_meta_multspec.spec_hw = best['spec_hw']
        s4_meta_multspec.p3thresh = best['p3thresh']
        s4_meta_multspec.median_thresh = best['median_thresh']
        s4_meta_multspec.window_len = best['window_len']
        s4_meta_multspec.p5thresh = best['p5thresh']
        s4_meta_multspec.p7thresh = best['p7thresh']
        s4_meta_multspec.drift_range = best['drift_range']
        s4_meta_multspec.highpassWidth = best['highpassWidth']
        s4_meta_multspec.sigma = best['sigma']
        s4_meta_multspec.box_width = best['box_width']
        # Stage 5
        s5_meta_multspec.diffthresh = best['diffthresh']
        s5_meta_multspec.bg_hw = best['bg_hw']
        s5_meta_multspec.spec_hw = best['spec_hw']
        s5_meta_multspec.p3thresh = best['p3thresh']
        s5_meta_multspec.median_thresh = best['median_thresh']
        s5_meta_multspec.window_len = best['window_len']
        s5_meta_multspec.p5thresh = best['p5thresh']
        s5_meta_multspec.p7thresh = best['p7thresh']
        s5_meta_multspec.drift_range = best['drift_range']
        s5_meta_multspec.highpassWidth = best['highpassWidth']
        s5_meta_multspec.sigma = best['sigma']
        s5_meta_multspec.box_width = best['box_width']

        if enable_exotic_ld is True:

            # Retrieve Values for Exotic-ld
            s4_meta_multspec.teff = planet_data['Teff']
            s4_meta_multspec.logg = planet_data['stellar_gravity']
            s4_meta_multspec.metallicity = planet_data['Fe/H']

            s5_meta_multspec.teff = planet_data['Teff']
            s5_meta_multspec.logg = planet_data['stellar_gravity']
            s5_meta_multspec.metallicity = planet_data['Fe/H']

            # Turn on compute_ld
            s4_meta_multspec.compute_ld = True

            # Turn on compute white
            s4_meta_multspec.compute_white = True

            # Specify 1D or 3D Grid Model
            s4_meta_multspec.exotic_ld_grid = '3D'

            # Path for exotic-ld ancillary files 
            s4_meta_multspec.exotic_ld_direc = exotic_ld_direc

            # Path for exotic-ld throughput file 
            s4_meta_multspec.exotic_ld_file = exotic_ld_file

            # Turn on use_generate_ld and enter paths for ld files
            s5_meta_multspec.use_generate_ld = use_generate_ld

            # Path for ld file (white) 
            s5_meta_multspec.ld_file = ld_file

            # Path for ld file (white) 
            s5_meta_multspec.ld_file_white = ld_file_white


        directory = s4_meta_multspec.topdir
        # directory = last_outputdir_S4
        if not os.path.exists(directory):
            os.makedirs(directory)


        # # Create Run String
        if i == 0:
            run_string = create_run_string(eventlabel, s3_meta_multspec.topdir + "DataAnalysis/HST/" + eventlabel + "/JointSpec/")

        ## Run Stage 3

        # Regular expression pattern to find "VisitXX-XX"
        # pattern = r"Visit\d+-\d+"
        # pattern = r"Visit\d+"
        pattern = r"Visit\d+(-\d+)?"

        # Replacement string, concatenating "Visit" with the new visit_nums
        replacement = "Visit" + visit_nums

        # Replacing "VisitXX-XX" in the filepath with "Visit" + visit_nums
        updated_inputdir = re.sub(pattern, replacement, s3_meta_multspec.inputdir)
        s3_meta_multspec.inputdir = updated_inputdir
        updated_inputdir_raw = re.sub(pattern, replacement, s3_meta_multspec.inputdir_raw)
        s3_meta_multspec.inputdir_raw = updated_inputdir_raw
        # print(updated_inputdir)
        # print(updated_inputdir_raw)

        s3_meta_multspec.outputdir_raw = "DataAnalysis/HST/" + eventlabel + "/JointSpec/" + run_string + "/S3/Visit" + visit_nums
        s3_meta_multspec.outputdir = s3_meta_multspec.topdir + s3_meta_multspec.outputdir_raw

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_multspec)
        last_outputdir_S3 = s3_meta.outputdir


        ## Run Stage 4 in loop for each spec chan
        s4_meta_multspec.inputdir = last_outputdir_S3

        outputdir = s4_meta_multspec.outputdir
        outputdir_raw = s4_meta_multspec.outputdir_raw

        # # Update outputdir to run-specific folder
        # # outputdir = outputdir + run_string + "/"
        # outputdir = s4_meta_multspec.topdir + "JointSpec/" + run_string + "/S4"

        # Define initial conditions
        wave_min_start = 1.12
        wave_max_start = 1.14
        increment = 0.02
        num_channels = 27

        for c in range(num_channels):
            # Update wave_min and wave_max
            s4_meta_multspec.wave_min = wave_min_start + (c * increment)
            s4_meta_multspec.wave_max = wave_max_start + (c * increment)

            # Split the string into parts
            parts = loc_sci.split('/')
            # Keep everything after the 3rd "/"
            visit_num = '/'.join(parts[-1:])

            # channel_dir = s4_meta_multspec.topdir + "/JointSpec/S4/chan" + str(c) + "/" + visit_num + "/"
            channel_dir = s4_meta_multspec.topdir + "/JointSpec/" + run_string + "/S4/chan" + str(c) + "/" + visit_num + "/"
            # channel_dir = outputdir + "/chan" + str(c) + "/" + visit_num + "/"
            s4_meta_multspec.outputdir = channel_dir

            # channel_dir_raw = "/JointSpec/S4/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            channel_dir_raw = "/JointSpec/" + run_string + "/S4/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            # channel_dir_raw = "/chan" + str(c) + "/" + visit_num + "/"   # Include path separator and start naming from chan1
            s4_meta_multspec.outputdir_raw = channel_dir_raw

            # Check if the directory exists, create it if it doesn't
            if not os.path.exists(channel_dir):
                os.makedirs(channel_dir)

            s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_multspec, s3_meta=s3_meta)

    run_dir = s4_meta_multspec.topdir + "/JointSpec/" + run_string + "/"

    return eventlabel, run_dir


def Stage5(eventlabel, run_dir, transits_to_mask):

    def count_matching_folders(start_dir, pattern):
        """
        Recursively count folders in start_dir matching the given pattern.

        Parameters:
        - start_dir (str): The directory to start the search from.
        - pattern (str): Regular expression pattern to match folder names.

        Returns:
        - int: The number of folders matching the pattern.
        """
        matching_count = 0

        # Compile the regular expression pattern for better performance
        compiled_pattern = re.compile(pattern)

        for root, dirs, _ in os.walk(start_dir):
            # Filter directories in 'dirs' that match the pattern
            matching_dirs = [d for d in dirs if compiled_pattern.match(d)]
            matching_count += len(matching_dirs)

        return matching_count

    # Define the pattern for matching folder names
    pattern_chan = r'^chan\d{1,2}$'
    # Count the matching folders
    matching_chan_count = count_matching_folders(run_dir+'S4/', pattern_chan)
    print(f"Number of matching channel folders: {matching_chan_count}")

    for c in range(matching_chan_count):
        search_directory = run_dir + 'S4/chan' + str(c)
        pattern_to_find = r'ap\d{2}_bg\d{2}'

        found_paths = []

        for root, dirs, files in os.walk(search_directory):
            for dir_name in dirs:
                if re.match(pattern_to_find, dir_name):
                    found_path = os.path.join(root, dir_name)
                    found_paths.append(found_path)

        # Dynamically remove the specified bad transits for the current channel
        # Reverse sorting to ensure deleting from the end doesn't affect indices of earlier items
        if len(transits_to_mask) > 0:
            for idx in sorted(transits_to_mask[c], reverse=True):
                if idx < len(found_paths):
                    del found_paths[idx]

        inputdir = found_paths[0]
        inputdirlist = found_paths[1:]


        # Define the pattern for matching folder names (VisitXX-XX)
        # pattern_visit = r'^Visit\d{2}-\d{2}$'
        # pattern_visit = r'^Visit\d{2}$'
        pattern_visit = r'^Visit\d{2}(-\d{2})?$'
        # Count the matching folders
        matching_visit_count = count_matching_folders(search_directory, pattern_visit)
        print(f"Number of matching visit folders: {matching_visit_count}")



        # Perform Joint Light Curve Fit

        eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

        ecf_path = '.'+os.sep
        s5_ecffile = 'S5_' + eventlabel + '.ecf'
        s5_meta_multspec = readECF.MetaClass(ecf_path, s5_ecffile)

        outputdir = run_dir + 'S5/chan' + str(c)

        parts = inputdir.split('/')
        # Keep everything after the 6th "/"
        inputdir_raw = '/'.join(parts[6:])

        parts = outputdir.split('/')
        # Keep everything after the 6th "/"
        outputdir_raw = '/'.join(parts[6:])

        s5_meta_multspec.multwhite = True
        s5_meta_multspec.inputdir = inputdir
        s5_meta_multspec.inputdirlist = inputdirlist
        s5_meta_multspec.inputdir_raw = inputdir_raw
        s5_meta_multspec.outputdir = outputdir 
        s5_meta_multspec.outputdir_raw = outputdir_raw

        # print(inputdir)
        # print(s5_meta_multspec.inputdir_raw)
        # print(inputdirlist)
        # print(outputdir)
        # print(s5_meta_multspec.outputdir_raw)

        # Set Fit Method
        s5_meta_multspec.fit_method = 'emcee'  

        s5_meta = s5.fitlc(eventlabel, ecf_path=ecf_path, s4_meta=None, input_meta=s5_meta_multspec) 
        # s5_meta = s5.fitlc(eventlabel, ecf_path=ecf_path, s4_meta=None)  # Shared

    def get_channel_number(dir_name):
        """
        Extracts the numerical part from a directory name formatted as 'chanX'.
        """
        try:
            return int(dir_name[4:])
        except ValueError:
            return -1  # Return -1 for any directory names that do not match expected format

    def process_images_in_figs_dir(figs_dir, chan_number):
        """
        Process all images in a given 'figs' directory within a channel folder and create a subplot compilation.
        """
        images = []
        image_files = [f for f in os.listdir(figs_dir) if f.endswith("_lc_emcee.png")]
        # Sorting the image files to ensure they are processed in order
        image_files.sort()
        for file in image_files:
            file_path = os.path.join(figs_dir, file)
            images.append(imread(file_path))
                
        if images:
            n = len(images)
            cols = 3
            rows = (n + cols - 1) // cols
            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            if n > 1:
                axs = axs.ravel()
            else:
                axs = [axs]
            for ax in axs[len(images):]:
                ax.axis('off')
            for img, ax in zip(images, axs):
                ax.imshow(img)
                ax.axis('off')
            plt.suptitle(f"Channel {chan_number}", fontsize=20)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            output_path = os.path.join(figs_dir, f"compilation_chan{chan_number}.png")
            plt.savefig(output_path)
            plt.close()
            print(f"Compilation saved: {output_path}")
        else:
            print(f"No images found in {figs_dir}")

    def find_and_process_figs_dir(run_dir):
        """
        Recursively search for 'chanX' directories to find 'figs' subdirectories and process the images within, ensuring channels are processed sequentially by channel number.
        """
        chan_dirs = []
        for root, dirs, files in os.walk(run_dir):
            for dir in dirs:
                if dir.startswith("chan"):
                    chan_dirs.append(os.path.join(root, dir))
        
        # Sort the channel directories by the numerical part of their names
        chan_dirs.sort(key=lambda x: get_channel_number(os.path.basename(x)))
        
        for chan_dir in chan_dirs:
            chan_number = get_channel_number(os.path.basename(chan_dir))
            if chan_number != -1:  # Process only if the folder name matches the expected format
                for sub_root, sub_dirs, sub_files in os.walk(chan_dir):
                    sub_dirs.sort()  # Sort sub_dirs to process them in order
                    if "figs" in sub_dirs:
                        figs_dir = os.path.join(sub_root, "figs")
                        process_images_in_figs_dir(figs_dir, str(chan_number))
                        break  # Assume only one 'figs' folder per channel folder

    # Example usage:
    find_and_process_figs_dir(run_dir)

    def extract_channel_number(filename):
        """
        Extract the channel number from the filename using a regular expression.
        Assumes filename format is 'compilation_chanX.png', where X is the channel number.
        """
        match = re.search(r'chan(\d+)', filename)
        return int(match.group(1)) if match else -1

    def create_pptx(run_dir, pptx_file_name):
        """
        Create a PowerPoint file with images sorted by their channel number,
        omitting the addition of text boxes.
        """
        prs = Presentation()
        blank_slide_layout = prs.slide_layouts[5]

        # Collect all relevant PNG files and their paths
        image_files = []
        for root, dirs, files in os.walk(run_dir):
            for file in files:
                if file.startswith("compilation_chan") and file.endswith(".png"):
                    full_path = os.path.join(root, file)
                    image_files.append(full_path)
        
        # Sort the files based on the channel number extracted from their filenames
        image_files.sort(key=lambda x: extract_channel_number(x))

        # Add sorted images to the presentation
        for img_path in image_files:
            slide = prs.slides.add_slide(blank_slide_layout)
            
            # Insert the image into the slide, adjusting the position and size as needed
            slide.shapes.add_picture(img_path, Inches(1), Inches(1), width=Inches(8))
        
        prs.save(run_dir + pptx_file_name)
        print(f"Presentation saved as {pptx_file_name}")

    # Example usage
    pptx_file_name = eventlabel + "_JointSpec_allChannels.pptx"
    create_pptx(run_dir, pptx_file_name)


def Stage6(eventlabel, run_dir):
    
    base_path = run_dir

    def find_log_files(base_path, eventlabel):
        """Search for .log files in the specified base_path."""
        return glob.glob(f"{base_path}**/S5_{eventlabel}.log", recursive=True)

    def extract_data_from_file(file_path):
        """Extract the necessary data from each log file."""
        with open(file_path, 'r') as file:
            content = file.read()
            bandpass_matches = re.findall(r'Bandpass \d+ = (\d+\.\d+) - (\d+\.\d+)', content)
            emcee_matches = re.findall(r'EMCEE RESULTS:\nrp: (\d+\.\d+) \(\+([-\d.e]+), -([-\d.e]+)\)', content)

            data = []
            data_ppm = []
            for bandpass in bandpass_matches:
                wave_lo, wave_hi = map(float, bandpass)
                wave = (wave_lo + wave_hi) / 2
                if emcee_matches:
                    rp, err_pos, err_neg = map(float, emcee_matches[0])
                    # rp_ppm = (rp ** 2) * 1e6  # Square rp to get the flux value, then convert to ppm
                    # err_pos_ppm = (((rp + err_pos) ** 2) - (rp ** 2)) * 1e6  # Convert positive error to ppm
                    # err_neg_ppm = abs((((rp - err_neg) ** 2) - (rp ** 2)) * 1e6)  # Convert negative error to ppm
                    # rp_ppm, err_pos_ppm, err_neg_ppm = rp, err_pos, err_neg
                    rp2 = rp ** 2  # Square rp to get the flux value, then convert to ppm
                    rp2_err_pos = abs(((rp + err_pos) ** 2) - (rp ** 2))  # Convert positive error to ppm
                    rp2_err_neg = abs(((rp - err_neg) ** 2) - (rp ** 2))  # Convert negative error to ppm
                    # data.append((wave, wave_lo, wave_hi, rp, err_pos, err_neg))
                    # data.append((wave, wave_lo, wave_hi, rp2, rp2_err_pos, rp2_err_neg))
                    data.append((wave, 0.01, rp2, rp2_err_pos, rp2_err_neg))  # 0.01 assumed for the 10nm bandpass halfwidth used for the WFC3 G141 spec channels (20nm spec channel width)
                    # data_ppm.append((wave, wave_lo, wave_hi, rp_ppm, err_pos_ppm, err_neg_ppm))
                    print("Data extracted from EMCEE file.")

            if not data:
                print("No EMCEE data found.")
            return data, data_ppm

    def plot_data(data, eventlabel):
        """
        Plot data with both x and y error bars, centering the y-axis data around 0.

        Parameters:
        - data: A list of tuples, each containing (wave, wave_lo, wave_hi, rp_ppm, err_pos_ppm, err_neg_ppm).
        """
        wave_list, wave_low_list, wave_high_list, rp_list, err_pos_list, err_neg_list = zip(*data)
        
        # Calculate the mean of rp_ppm values and subtract it from each rp_ppm to center around 0
        # mean_rp = np.mean(rp_list)
        mean_rp = 0
        adjusted_rp_list = [rp - mean_rp for rp in rp_list]
        
        xerrors = [[wave - low for wave, low in zip(wave_list, wave_low_list)], 
                [high - wave for wave, high in zip(wave_list, wave_high_list)]]

        plt.figure(figsize=(16, 4))
        plt.errorbar(wave_list, adjusted_rp_list, xerr=xerrors, yerr=[err_neg_list, err_pos_list], fmt='o', capsize=5, 
                    ecolor='darkorange', marker='s', mfc='skyblue', mec='darkblue', 
                    linestyle='None', markersize=6, elinewidth=2)
        
        plt.xlabel('Wavelength (μm)', fontsize=14)
        plt.ylabel('Transit Depth (ppm)', fontsize=14)
        # plt.ylabel('Relative Transit Depth (ppm)', fontsize=14)
        plt.title(eventlabel, fontsize=18)

        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.xlim(1.1, 1.7)

        save_dir = base_path + 'S6/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(save_dir + 'TransmissionSpectra_' + eventlabel + '_JointSpec.pdf')

        plt.show()

    def plot_data_interactive(data, eventlabel):
        """
        Plot data interactively with both x and y error bars, centering the y-axis data around 0.
        """
        wave_list, wave_low_list, wave_high_list, rp_list, err_pos_list, err_neg_list = zip(*data)

        # Error in wavelength (x-axis)
        x_errors = [[-0.01 for wave, low in zip(wave_list, wave_low_list)], 
                    [0.01 for wave, high in zip(wave_list, wave_high_list)]]

        # Error in transit depth (y-axis)
        y_errors = [err_neg_list, err_pos_list]

        fig = go.Figure()

        for x, y, x_err, y_err in zip(wave_list, rp_list, zip(*x_errors), zip(*y_errors)):
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                error_x=dict(type='data', array=[x_err[1]], arrayminus=[x_err[0]]),
                error_y=dict(type='data', array=[y_err[1]], arrayminus=[y_err[0]]),
                mode='markers',
                marker=dict(size=10, color='skyblue', line=dict(width=2, color='DarkSlateGrey')),
                name=''
            ))

        fig.update_layout(
            title=eventlabel,
            xaxis_title='Wavelength (μm)',
            yaxis_title='Transit Depth (ppm)',
            xaxis=dict(range=[1.1, 1.7]),
            template='plotly_white'
        )

        fig.show()

    # You would call plot_data_interactive in your main function instead of plot_data


    def save_data_to_file(data, file_path):
        """Write the data to a .txt file."""
        with open(file_path, 'w') as f:
            f.write(
'''
# ECSV 1.0
# ---
# datatype:
# - name: wavelength, datatype: float64
# - name: bin_width, datatype: float64
# - name: rp^2_value, datatype: float64
# - name: rp^2_errorneg, datatype: float64
# - name: rp^2_errorpos, datatype: float64
# schema: astropy-2.0
wavelength bin_width rp^2_value rp^2_errorneg rp^2_errorpos
'''
                )

            for item in data:
                f.write(" ".join(map(str, item)) + "\n")

    # def main():
    #     log_files = find_log_files(base_path, eventlabel)
    #     all_data = []

    #     for file_path in log_files:
    #         data = extract_data_from_file(file_path)
    #         all_data.extend(data)

    #     if all_data:
    #         # Convert list to numpy array for easier sorting and manipulation
    #         all_data_array = np.array(all_data, dtype=float)
    #         # Sort by wavelength
    #         sorted_all_data = sorted(all_data_array, key=lambda x: x[0])
            
    #         # Identify unique wavelengths and exclude the last two
    #         unique_wavelengths = sorted(set(item[0] for item in sorted_all_data))
    #         if len(unique_wavelengths) > 1:
    #             # Exclude the last two unique wavelengths
    #             wavelengths_to_keep = unique_wavelengths[:-1]
    #             filtered_data = [item for item in sorted_all_data if item[0] in wavelengths_to_keep]
    #         else:
    #             filtered_data = sorted_all_data
            
    #         # Proceed with filtered data
    #         rp_list = [item[3] for item in filtered_data]  # Extracting rp_ppm values
    #         mean_rp_ppm = np.mean(rp_list)
    #         adjusted_rp_list = [(item[0], item[1], item[2], item[3] - mean_rp_ppm, item[4], item[5]) for item in filtered_data]
            
    #         plot_data(adjusted_rp_list, eventlabel)
    #         # Assuming you want to save the adjusted data to a file
    #         save_data_to_file(adjusted_rp_list, 'AdjustedTransmissionSpectra_' + eventlabel + '.txt')
    #         print("Data plotted and adjusted data saved, excluding the last two channels.")
    #     else:
    #         print("No data found.")

    def main():
        log_files = find_log_files(base_path, eventlabel)
        all_data = []
        # all_data_ppm = []

        for file_path in log_files:
            data, data_ppm = extract_data_from_file(file_path)
            all_data.extend(data)
            # all_data_ppm.extend(data_ppm)

        if all_data:
            # Convert list to numpy array for easier sorting and manipulation
            all_data_array = np.array(all_data, dtype=float)
            sorted_all_data = sorted(all_data_array, key=lambda x: x[0])  # Sort by wavelength

            # all_data_ppm_array = np.array(all_data_ppm, dtype=float)
            # sorted_all_data_ppm = sorted(all_data_ppm_array, key=lambda x: x[0])  # Sort by wavelength

            save_dir = base_path + 'S6/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_data_to_file(sorted_all_data, save_dir + 'TransmissionSpectra_' + eventlabel + '_JointSpec.txt')

            # plot_data_interactive(sorted_all_data_ppm, eventlabel)
            plot_data_interactive(sorted_all_data, eventlabel)

            # sorted_all_data_ppm = sorted_all_data_ppm[1:]
            sorted_all_data = sorted_all_data[1:]

            # plot_data(sorted_all_data_ppm, eventlabel)
            plot_data(sorted_all_data, eventlabel)

            print("Data plotted and saved.")

        else:
            print("No data found.")

    # if __name__ == "__main__":
    main()