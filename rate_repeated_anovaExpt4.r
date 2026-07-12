# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

library(tidyverse)
library(ez)
library(rstatix)
library(lmerTest)

moving_df <- read.csv('temp/rate_no_exclusion')
rstatix_aov <- anova_test(Rate ~ Phasic * Tonic + Error(Subjects / (Phasic * Tonic)), data=moving_df)
print(rstatix_aov)

ez_aov <- ezANOVA(data=moving_df, dv=Rate, wid=Subjects, within=list(Phasic, Tonic))
print(ez_aov)