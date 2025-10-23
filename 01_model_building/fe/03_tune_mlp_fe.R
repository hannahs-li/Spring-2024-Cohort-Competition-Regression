# Regression Prediction Problem ----
# Single layer neural net tuning, fe imputation ----

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
load(here("recipes/recipe_fe.rda"))

# model specifications ----
mlp_spec <- mlp(
  mode = "regression",
  hidden_units = tune(),
  penalty = tune()
) |> 
  set_engine("nnet")

# define workflow ----
mlp_wflow <- workflow() |> 
  add_model(mlp_spec) |> 
  add_recipe(recipe_fe)

# hyperparameter tuning values ----
mlp_param <- hardhat::extract_parameter_set_dials(mlp_spec)

mlp_grid <- grid_latin_hypercube(mlp_param, size = 50)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("MLP: FE") # start clock

# tuning code in here
tune_mlp_fe <- mlp_wflow |> 
  tune_grid(
    resamples = folds, 
    grid = mlp_grid,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_mlp_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

stopCluster(cl)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_mlp_fe, file = here("results/tune_mlp_fe.rda"))
save(tictoc_mlp_fe, file = here("results/tictoc_mlp_fe.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_mlp_fe |> 
  extract_workflow(tune_mlp_fe) |>  
  finalize_workflow(select_best(tune_mlp_fe, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_mlp_fe <- fit(final_wflow, train_train)

metrics_mlp_fe <- bind_cols(train_test, predict(final_tune_mlp_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_mlp_fe, file = here("results/metrics/metrics_mlp_fe.rda"))


