## Regression Prediction Problem ----
## Data Split and Fold ----

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts 
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))

# folding data (resamples) ----
# set seed 
set.seed(1234)

split <- train |> 
  initial_split(prop = 0.75,strata = price_log10)

train_train <- training(split)
train_test <- testing(split)

folds <- train_train |> 
  vfold_cv(v = 5, repeats = 3, strata = price_log10)

# save ----
save(folds, file = here("data/folds.rda"))
save(train_train, file = here("data/train_train.rda"))
save(train_test, file = here("data/train_test.rda"))
