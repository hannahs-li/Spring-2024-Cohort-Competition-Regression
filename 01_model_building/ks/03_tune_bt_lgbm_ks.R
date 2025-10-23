# Regression Prediction Problem ----
# Define and fit bt model on ks recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)
library(bonsai)

# handle common conflicts
tidymodels_prefer()

# set parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load training data
load(here("data/folds.rda"))
load(here("data/train.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks.rda"))

# model specs ----
bt_spec <- boost_tree(
  mode = "regression",
  mtry = tune(),
  learn_rate = tune(),
  min_n = tune(),
  trees = 1000,
  tree_depth = tune()
) |>
  set_engine("lightgbm")

# workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(mtry = mtry(c(18, 25))) |> 
  update(learn_rate = learn_rate(c(-1.6, -1.4))) |> 
  update(min_n = min_n(c(2, 10))) |> 
  update(tree_depth = tree_depth(c(6, 7)))

bt_grid <- grid_regular(bt_params, levels = c(3, 2, 2, 8))

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("BT LGBM: KS") # start clock

# tuning code in here
tune_lgbm_bt_ks <- tune_grid(
  bt_wflow, 
  resamples = folds,
  grid = bt_grid,
  metrics = metric_set(mae),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_lgbm_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_lgbm_bt_ks, file = here("results/tune_lgbm_bt_ks.rda"))
save(tictoc_bt_lgbm_ks, file = here("results/tictoc_bt_lgbm_ks.rda"))

load(here("results/tune_lgbm_bt_ks.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

autoplot(tune_lgbm_bt_ks)
show_best(tune_lgbm_bt_ks, metric = "mae")

# collect metrics ----
# extract final workflow 
final_wflow <- tune_lgbm_bt_ks |> 
  extract_workflow(tune_lgbm_bt_ks) |>  
  finalize_workflow(select_best(tune_lgbm_bt_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_bt_lgbm_ks <- fit(final_wflow, train_train)

metrics_bt_lgbm_ks <- bind_cols(train_test, predict(final_tune_bt_lgbm_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_bt_lgbm_ks, file = here("results/metrics/metrics_bt_lgbm_ks.rda"))
