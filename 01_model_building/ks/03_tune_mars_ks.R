# Regression Prediction Problem ----
# Tuning for MARS model with ks ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load data  ----
load(here("data/folds.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# load preprocessing/recipe ----
load(here("recipes/recipe_ks.rda"))

# model specifications ----
mars_spec <- mars(
  num_terms = tune(),
  prod_degree = tune()
) |> 
  set_mode("regression") |> 
  set_engine("earth")

# define workflow ----
mars_wflow <- workflow() |> 
  add_model(mars_spec) |> 
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
mars_params <- hardhat::extract_parameter_set_dials(mars_spec) |> 
  update(
    num_terms = num_terms(range = c(1L, 25L))
  )

mars_grid <- grid_regular(mars_params, levels = 24)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Mars with KS") # start clock

tune_mars_ks <- tune_grid(
  mars_wflow, 
  resamples = folds,
  grid = mars_grid,
  metrics = metric_set(mae),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_mars_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_mars_ks, file = here("results/tune_mars_ks.rda"))
save(tictoc_mars_ks, file = here("results/tictoc_mars_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_mars_ks |> 
  extract_workflow(tune_mars_ks) |>  
  finalize_workflow(select_best(tune_mars_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_mars_ks <- fit(final_wflow, train_train)

metrics_mars_ks <- bind_cols(train_test, predict(final_tune_mars_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_mars_ks, file = here("results/metrics/metrics_mars_ks.rda"))
