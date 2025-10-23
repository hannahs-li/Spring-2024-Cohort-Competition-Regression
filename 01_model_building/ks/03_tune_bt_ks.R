# Regression Prediction Problem ----
# Define and fit bt model on ks recipe ---

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

# model specs ----
bt_spec <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune()
) |>
  set_engine("xgboost")

# workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(mtry = mtry(c(1, 7))) |> 
  update(learn_rate = learn_rate(c(-5, -0.2)))

bt_grid <- grid_regular(bt_params, levels = 5)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("BT with KS") # start clock

# tuning code in here
tune_bt_ks <- tune_grid(
  bt_wflow, 
  resamples = folds,
  grid = bt_grid,
  metrics = metric_set(mae),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_bt_ks, file = here("results/tune_bt_ks.rda"))
save(tictoc_bt_ks, file = here("results/tictoc_bt_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_bt_ks |> 
  extract_workflow(tune_bt_ks) |>  
  finalize_workflow(select_best(tune_bt_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_bt_ks <- fit(final_wflow, train_train)

metrics_bt_ks <- bind_cols(train_test, predict(final_tune_bt_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_bt_ks, file = here("results/metrics/metrics_bt_ks.rda"))