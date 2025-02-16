# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
gsr_df <- read.csv(args[1])

m <- glm(formula = GSR ~ Seen_Green + Pick_Green, data = gsr_df, na.action=na.omit)

sink(args[2])
summary(m)
sink()