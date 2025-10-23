# Regression Prediction Problem ----
# Define and fit svm poly model on ks recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(kernlab)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# set parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load training data
load(here("data/folds.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks.rda"))

# model specifications ----
svm_poly_spec <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) |>
  set_engine("kernlab") |>
  set_mode("regression")

# define workflows ----
svm_poly_wflow <- workflow() |>
  add_model(svm_poly_spec) |>
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
svm_poly_params <- hardhat::extract_parameter_set_dials(svm_poly_spec)

# define grid
svm_poly_grid <- grid_latin_hypercube(svm_poly_params, size = 53)

# fit workflow/model ----
tic("SVM Poly KS") # start clock

tune_svm_poly_ks <- 
  svm_poly_wflow |>
  tune_grid(
    resamples = folds,
    metrics = metric_set(mae),
    grid = svm_poly_grid,
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_poly_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_poly_ks, file = here("results/tune_svm_poly_ks.rda"))
save(tictoc_svm_poly_ks, file = here("results/tictoc_svm_poly_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_svm_poly_ks |> 
  extract_workflow(tune_svm_poly_ks) |>  
  finalize_workflow(select_best(tune_svm_poly_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_svm_poly_ks <- fit(final_wflow, train_train)

metrics_svm_poly_ks <- bind_cols(train_test, predict(final_tune_svm_poly_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_svm_poly_ks, file = here("results/metrics/metrics_svm_poly_ks.rda"))