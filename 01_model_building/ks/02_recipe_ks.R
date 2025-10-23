# Regression ----
# Setup ks preprocessing/recipes

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data ----
load(here("data/train.rda"))

# ks recipe ----
# build recipe 
recipe_ks <- recipe(price_log10 ~., data = train) |>
  step_rm(id) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_nzv(all_numeric_predictors()) |> 
  step_normalize(all_predictors())

# check recipe 
recipe_ks |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

# save recipe ----
save(recipe_ks, file = here("recipes/recipe_ks.rda"))
