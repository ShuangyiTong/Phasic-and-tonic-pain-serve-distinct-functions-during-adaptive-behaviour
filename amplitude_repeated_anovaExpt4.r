# Copyright (c) 2024 Shuangyi Tong <shuangyi.tong@eng.ox.ac.uk>
# Licensed under the MIT License (SPDX: MIT).

library(tidyverse)
library(ez)
library(rstatix)
library(lmerTest)

df <- read.csv('temp/rate_no_exclusion')
rstatix_aov <- anova_test(Amplitude ~ Phasic * Tonic + Error(Subjects / (Phasic * Tonic)), data=df)
print(rstatix_aov)

ez_aov <- ezANOVA(data=df, dv=Amplitude, wid=Subjects, within=list(Phasic, Tonic))
print(ez_aov)