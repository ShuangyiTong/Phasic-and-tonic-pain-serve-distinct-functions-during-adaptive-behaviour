# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

import sys
import cairosvg
import io
import numpy as np
from PIL import Image
import cv2

def svgRead(filename, dpi=96):
   """Load an SVG file and return image in Numpy array"""
   # Make memory buffer
   mem = io.BytesIO()
   # Convert SVG to PNG in memory
   cairosvg.svg2png(url=filename, write_to=mem, dpi=dpi)
   # Convert PNG to Numpy array
   return np.array(Image.open(mem))

stitch_target = sys.argv[1]

if stitch_target == 'gsr':
    figure1 = 'figures/PUB/gsr.svg'
    figure2 = 'figures/PUB/seen_coef.svg'

    figure1_raster = svgRead(figure1)
    figure2_raster = svgRead(figure2)

    f1_box = figure1_raster.shape
    f2_box = figure2_raster.shape
    print(f1_box)
    print(f2_box)

    canvas = np.zeros((2100, 2000, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[20:20 + f1_box[0], 150:150 + f1_box[1]] = cv2.cvtColor(figure1_raster, cv2.COLOR_RGB2BGR)
    canvas[f1_box[0] + 100:f1_box[0] + 100 + f2_box[0], 10:10 + f2_box[1]] = cv2.cvtColor(figure2_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(i)", (30, 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = cv2.putText(canvas, "(ii)", (30, 1050), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite("figures/PUB/stitched_gsr.png", canvas)
    
elif stitch_target == 'expt2':
    figure1 = 'figures/PUB/Ratings_vs_pain_choice_prob_lmerExpt2.svg'
    figure2 = 'figures/PUB/Ratings_vs_distance_bias_lmerExpt2.svg'
    figure3 = 'figures/PUB/Expt2_Model_Fitting_Coefficients.svg'
    figure4 = 'figures/PUB/Phasic_Pain_Utility_Function_To_Ratings.svg'

    figure1_raster = svgRead(figure1, dpi=150)
    figure2_raster = svgRead(figure2, dpi=150)
    figure3_raster = svgRead(figure3)
    figure4_raster = svgRead(figure4)

    f1_box = figure1_raster.shape
    f2_box = figure2_raster.shape
    print(f1_box)
    print(f2_box)
    f3_box = figure3_raster.shape
    f4_box = figure4_raster.shape
    print(f3_box)
    print(f4_box)

    canvas = np.zeros((3050, 2760, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[40:40 + f1_box[0], 20:20 + f1_box[1]] = cv2.cvtColor(figure1_raster, cv2.COLOR_RGB2BGR)
    canvas[40:40 + f2_box[0], 60 + f1_box[1]:60 + f1_box[1] + f2_box[1]] = cv2.cvtColor(figure2_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(i)", (30, 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = cv2.putText(canvas, "(ii)", (30 + f2_box[1], 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)

    canvas[900 + 40:900 + 40 + f3_box[0], 250:250 + f3_box[1]] = cv2.cvtColor(figure3_raster, cv2.COLOR_RGB2BGR)
    canvas[f3_box[0] + 40 + 900:f3_box[0] + 40 + 900 + f4_box[0], 120:120 + f4_box[1]] = cv2.cvtColor(figure4_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(iii)", (30, 1000), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite("figures/PUB/stitched_expt2_fit.png", canvas)

elif stitch_target == 'tonic_no_effect':
    figure1 = 'figures/PUB/Ratings_by_conditions_expt4.svg'
    figure2 = 'figures/PUB/choice_probability_tonic.svg'

    figure1_raster = svgRead(figure1)
    figure2_raster = svgRead(figure2)    
    
    f1_box = figure1_raster.shape
    f2_box = figure2_raster.shape
    print(f1_box)
    print(f2_box)

    canvas = np.zeros((1700, 2150, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[20:20 + f1_box[0], 70:70 + f1_box[1]] = cv2.cvtColor(figure1_raster, cv2.COLOR_RGB2BGR)
    canvas[f1_box[0] + 100:f1_box[0] + 100 + f2_box[0], 70:70 + f2_box[1]] = cv2.cvtColor(figure2_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(i)", (20, 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = cv2.putText(canvas, "(ii)", (20, 1050), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite("figures/PUB/stitched_tonic_no_effect.png", canvas)

elif stitch_target == 'topo':
    figure1 = 'figures/PUB/time_frequency_topo_tonic.svg'
    figure2 = 'figures/PUB/time_frequency_topo_vigour_constant.svg'

    figure1_raster = svgRead(figure1)
    figure2_raster = svgRead(figure2)    
    
    f1_box = figure1_raster.shape
    f2_box = figure2_raster.shape
    print(f1_box)
    print(f2_box)

    canvas = np.zeros((2100, 3300, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[40:40 + f1_box[0], 0:0 + f1_box[1]] = cv2.cvtColor(figure1_raster, cv2.COLOR_RGB2BGR)
    canvas[f1_box[0] + 100:f1_box[0] + 100 + f2_box[0], 0:0 + f2_box[1]] = cv2.cvtColor(figure2_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(i) Tonic pain conditions", (100, 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = cv2.putText(canvas, "(ii) Fitted vigour constants", (100, 1100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite("figures/PUB/stitched_topo.png", canvas)

elif stitch_target == 'expt4_fit':
    figure1 = 'figures/PUB/Expt4_vigour_fit.svg'
    figure2 = 'figures/PUB/Expt4_phasic_fitting.svg'

    figure1_raster = svgRead(figure1)
    figure2_raster = svgRead(figure2)

    f1_box = figure1_raster.shape
    f2_box = figure2_raster.shape
    print(f1_box)
    print(f2_box)

    canvas = np.zeros((1900, 3000, 3), dtype=np.uint8)
    canvas.fill(255)
    canvas[20:20 + f1_box[0], 450:450 + f1_box[1]] = cv2.cvtColor(figure1_raster, cv2.COLOR_RGB2BGR)
    canvas[f1_box[0] + 100:f1_box[0] + 100 + f2_box[0], 10:10 + f2_box[1]] = cv2.cvtColor(figure2_raster, cv2.COLOR_RGB2BGR)
    canvas = cv2.putText(canvas, "(i)", (300, 100), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    canvas = cv2.putText(canvas, "(ii)", (30, 850), cv2.FONT_HERSHEY_DUPLEX , 1.75, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.imwrite("figures/PUB/stitched_expt4_fit.png", canvas)