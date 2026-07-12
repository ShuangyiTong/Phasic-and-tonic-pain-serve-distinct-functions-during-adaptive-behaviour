# Code and data for the paper: Phasic and tonic pain serve distinct functions during adaptive behaviour

Contact person: Shuangyi Tong \<shuangyi.tong@eng.ox.ac.uk>. Alternative personal email: \<s9tong@edu.uwaterloo.ca>

Corresponding publication: https://elifesciences.org/reviewed-preprints/107911v2

Go to https://doi.org/10.5281/zenodo.21327818 to download full data.

## System Requirements

**System**: Windows 11 23H2, Intel CPU. For AMD CPU results, see supplementary materials.

**Python version**: Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32

Required packages are listed in `requirements.txt`. As of February 2025, the only incompatibility with up-to-date packages is `pandas`, which has deprecated the `append` function. You therefore need to install the specific version of `pandas` listed in `requirements.txt`, or, if you prefer a newer version of `pandas`, modify the code to use `concat` instead.

You will also need to install `cairosvg` if you intend to merge figures as described in the steps below. This is only for publication aesthetics, so the merge steps can be safely skipped.

**R version**: R version 4.4.2 (2024-10-31 ucrt) -- "Pile of Leaves", x86_64-w64-mingw32/x64

The following R packages are required: `lmerTest`, `ggplot2`, `tidyverse`, `lazyeval`, `rstatix`, `car`, `broom.mixed`, `ez`, `generics`. Make sure `rscript` has been added to your `PATH` environment variable.

## Figure-by-figure instructions

Open a terminal in this folder (as the current working directory) and run the following commands to generate the corresponding figures.

**Figure 1**: 

```
python plot_trajectory_heatmap.py
```

**Figure 3**: 

Panel A
```
python ratings_vs_pain_choice_bias.py
```
Panel B
```
python distance_compare.py Expt2
```
Panel CDEFG
```
# First fit the model
python old_model_fast.py Expt2
# Then plot the fitting results
python Expt2_modelling_fitting_plots.py
```
Merge all panels into one single figure
```
python stitch_helper.py expt2
```

**Figure 4**:

Panel A
```
python GSR_plots.py Expt2
```
Panel BC
```
python GSR_fit.py Expt2
```
Merge all panels into one single figure
```
python stitch_helper.py gsr
```

**Figure 5**:

Panel AB
```
python pain_condition_vs_ratings.py
```
Panel CDE
```
python choice_probability_tonic_vs_no_tonic.py
```
Merge all panels into one single figure
```
python stitch_helper.py tonic_no_effect
```

**Figure 6**:
```
python pain_erp.py
```

**Figure 7**
```
python moving_speed_trajectory_collection_rate.py
```

**Figure 9**:

Fitted results are provided in `model_fitting_results`. If you wish to run the model-fitting yourself, please refer to the model-fitting section below.

Panel AB
```
python Expt4_model_fitting_plot_vigour.py
```
Panel CDE
```
python Expt4_model_fitting_plot_phasic.py
```
Merge all panels into one single figure
```
python stitch_helper.py expt4_fit
```

**Figure 10**:

Generate spectral fit
```
python eeg_with_lmm.py Expt4 surface
```
Panel A
```
python topography_replot.py
```
Panel B (rerun after changing `quantities = 'tonic'` to `quantities = 'vigour_constant'`)
```
python topography_replot.py
```
Merge all panels into one single figure
```
python stitch_helper.py topo
```

**Source analysis in Supplementary**

Run
```
python eeg_with_lmm.py Expt4 surface_source
```
The script contains many options reflecting the different methods and settings we explored; however, we recommend using only the values already set in the script. Some options that appear configurable were not used for the published figures and may be obsolete.

Then run the following to generate the combined plots
```
python surface_source_plot.py
```

## Vigour-opportunity cost model-fitting

To understand the code, `realtime_model.py` is sufficient. `model_validation.py` (prediction only) and `model_fitting_results_compare.py` (which also includes the grid-search algorithm) verify that the CUDA / OpenMP implementation is identical to the Python version, up to a small floating-point error arising from the use of FP32 on CUDA.

To fit the model, you need either a reasonably capable CUDA-compatible GPU (we used a single RTX 3090, by 2024 standards) or at least several hundred CPU cores configured to run OpenMP, so that fitting completes within a few days. The CPU OpenMP version has not been fully tested (although the code has been written), so we provide instructions for the CUDA version only.

### Build executable

**Install the CUDA toolkit**: We used version 12.4 (Cuda compilation tools, release 12.4, V12.4.131, Build cuda_12.4.r12.4/compiler.34097967_0).

**Change the CUDA capability flag in the Makefile**: In the model-fitting folder, open the Makefile and replace `-arch=sm_86` with the flag appropriate for your GPU (no change is needed if you are also using an NVIDIA RTX 3090).

**Run Makefile**:
```
make global_optimizer_cuda
```

### Prepare data

The CUDA implementation relies on data dumped by a Python script to speed up computation (for example, to avoid string comparisons in CUDA). The tonic / no-tonic pain conditions are configured via `realtime_model.NO_CONDITION_1` and `realtime_model.NO_CONDITION_2`. Setting `realtime_model.NO_CONDITION_1=False` and `realtime_model.NO_CONDITION_2=True` dumps only the tonic pain-condition blocks; setting the opposite values dumps the no-tonic pain-condition blocks. Run the following script to dump the data for CUDA:
```
python dump_realtime_behavioural_data.py Expt4
```

### Run the model-fitting

The dumped data is located in `temp/behavioural_dump_json_Expt4`. You can now run the model-fitting:
```
.\global_optimizer_cuda.exe temp/behavioural_dump_json_Expt4
```
A JSON file will be generated as the output of the model-fitting. You can verify this output against the corresponding file in the `model_fitting_results` folder; the two should be identical.