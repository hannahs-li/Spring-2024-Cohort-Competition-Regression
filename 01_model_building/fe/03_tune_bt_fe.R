# Regression Prediction Problem ----
# Define and fit bt model on fe recipe

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(xgboost)
library(tictoc)

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
load(here("recipes/recipe_fe.rda"))

# model specs ----
bt_spec <- boost_tree(
  mode = "regression",
  mtry = tune(),
  min_n = tune(),
  learn_rate = tune(),
  trees = 1000,
  tree_depth = tune()
) |>
  set_engine("xgboost")

# workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(recipe_fe)

# hyperparameter tuning values ----
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(mtry = mtry(c(15, 20))) |> 
  update(learn_rate = learn_rate(c(-1.6, -1.3))) |> 
  update(tree_depth = tree_depth(c(5, 7)))

bt_grid <- grid_regular(bt_params, levels = c(2, 2, 3, 5))

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("BT with FE") # start clock

# tuning code in here
tune_bt_fe <- tune_grid(
  bt_wflow, 
  resamples = folds,
  grid = bt_grid,
  metrics = metric_set(mae),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_bt_fe, file = here("results/tune_bt_fe.rda"))
save(tictoc_bt_fe, file = here("results/tictoc_bt_fe.rda"))

# load(here("results/tune_bt_fe.rda"))
# autoplot(tune_bt_fe)
# show_best(tune_bt_fe)

# collect metrics ----
# extract final workflow 
final_wflow <- tune_bt_fe |> 
  extract_workflow(tune_bt_fe) |>  
  finalize_workflow(select_best(tune_bt_fe, metric = "mae"))

# train final model 
set.seed(1234)
final_fit_bt_fe <- fit(final_wflow, train_train)

metrics_bt_fe <- bind_cols(train_test, predict(final_fit_bt_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)
 
# save model metrics ----
save(metrics_bt_fe, file = here("results/metrics/metrics_bt_fe.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_bt <- bind_cols(test, predict(final_fit_bt_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted)

# save submission test ----
write_csv(submission_test_bt, file = here("submissions/submission_test_bt.csv"))
