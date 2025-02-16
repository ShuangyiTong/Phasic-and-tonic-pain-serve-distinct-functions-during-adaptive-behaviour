Codebase for https://www.biorxiv.org/content/10.1101/2025.02.10.637253v1

Dataset too large, so full data has not been uploaded to Github at the moment

# How to reproduce figures from raw data

Contact person: Shuangyi Tong \<shuangyi.tong@eng.ox.ac.uk>. Alternative emails: \<s9tong@edu.uwaterloo.ca>, \<shuangyi.tong@eng.ox.ac.uk>.

## System Requirements

**System**: Windows 11 23H2, Intel CPU. For AMD CPU results, see supplementary materials.

**Python version**: Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32

Required packages see `requirements.txt`. As of Feburary 2025, the only issue with up-to-date packages is the `pandas` package which they deprecated `append` function. So you need to install the specific version of `pandas` listed in the `requirements.txt`. Or you can fix the code by using `concat` function if you prefer newer pandas.

**R version**: R version 4.4.2 (2024-10-31 ucrt) -- "Pile of Leaves", x86_64-w64-mingw32/x64

Key user package required: `lmerTest`, `ggplot2`, `tidyverse`, `lazyeval`, `rstatix`, `car`, `broom.mixed`, `ez`, `generics`.

## Steps by figures

Open a terminal window with this folder as the current working directory and type the following commands to generate corresponding figures

**Figure 1**: 

```
python plot_trajectory_heatmap.py
```

**Figure 2**: 

Panel (i) 
```
python ratings_vs_pain_choice_bias.py
```
Panel (ii)
```
python distance_compare.py Expt2
```
Panel (iii)
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

**Figure 3**:

Panel (i)
```
python GSR_plots.py Expt2
```
Panel (ii)
```
python GSR_fit.py Expt2
```
Merge all panels into one single figure
```
python stitch_helper.py gsr
```

**Figure 4**:

Panel (i)
```
python pain_condition_vs_ratings.py
```
Panel (ii)
```
python choice_probability_tonic_vs_no_tonic.py
```
Merge all panels into one single figure
```
python stitch_helper.py tonic_no_effect
```

**Figure 5**:
```
python pain_erp.py
```

**Figure 6**
```
python moving_speed_trajectory_collection_rate.py
```

**Figure 7**:

Fitted results have been placed under model_fitting_results. If you want to run the model-fitting yourself, please refer to the model-fitting section. 

Panel (i)
```
python Expt4_model_fitting_plot_vigour.py
```
Panel (ii)
```
python Expt4_model_fitting_plot_phasic.py
```
Merge all panels into one single figure
```
python stitch_helper.py expt4_fit
```

**Figure 8**:

Generate spectral fit
```
python eeg_with_lmm.py Expt4 surface
```
Panel (i)
```
python topography_replot.py
```
Panel (ii) (rerun by change quantities = 'tonic' to quantities = 'vigour_constant')
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
There are many options as I was trying different methods and settings, but we only recommend whatever was set in the script. Some implementation that looks like configurable but not used in published figures may be obsolete and not recommended.

Then run the following to generate the combined plots
```
python surface_source_plot.py
```

## Vigour-opportunity cost model-fitting

If you want to understand the code, `realtime_mode.py` suffices, and `model_validation.py` (prediction only) and `model_fitting_results_compare.py` (also includes grid search algorithm) verifies the implementation in CUDA / OpenMP is identical to the Python version (up to some small floating point error as we use FP32 on CUDA). 

To fit the model, you need one decent GPU (by 2024 standard, we were using a single RTX 3090) that supports CUDA or you have at least hundreds of CPU cores configured to run OpenMP so that it completes within a few days. The CPU OpenMP version has not been finally tested (although code has been written), so we only have instructions for CUDA version here.

### Build executable

**Install CUDA toolkit**: We were using 12.4 version. Cuda compilation tools, release 12.4, V12.4.131, Build cuda_12.4.r12.4/compiler.34097967_0

**Change CUDA capability flag in the Makefile**: Go to model-fitting folder, find the Makefile, find `-arch=sm_86`, replace it with the correct flag (no need to change if you also use NVIDIA RTX 3090).

**Run Makefile**:
```
make global_optimizer_cuda
```

### Prepare data

The CUDA implementation relies on Python script dumped data to speed up computation (e.g. you don't want to compare strings in CUDA). The tonic / no tonic pain conditions is configured by `realtime_model.NO_CONDITION_1` and `realtime_model.NO_CONDITION_2`. Set `realtime_model.NO_CONDITION_1=False` and `realtime_model.NO_CONDITION_2=True` only dump tonic pain condition blocks. Set opposite value to generate no tonic pain condition blocks dump. Run the following script to dump data for CUDA
```
python dump_realtime_behavioural_data.py Expt4
```

### Run the model-fitting

The dumped folder locates in `temp/behavioural_dump_json_Expt4`. We can now run the model-fitting
```
.\global_optimizer_cuda.exe temp/behavioural_dump_json_Expt4
```
A Json file will be generated as the output of the model-fitting. You can verify the output Json file against the one in `model-fitting results` folder, and they should be identical.