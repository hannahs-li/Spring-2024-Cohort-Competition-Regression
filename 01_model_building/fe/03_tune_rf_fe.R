# Regression Prediction Problem ----
# Define and fit rf model with fe recipe ----

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(earth)
library(tictoc)

# load training data 
load(here("data/folds.rda"))
load(here("data/train.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# make rf data/remove na ----
rf_folds <- train_train |> drop_na() |> vfold_cv(v = 5, repeats = 3, strata = price_log10)

# load recipe 
load(here("recipes/recipe_rf.rda"))

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE) 
registerDoMC(cores = num_cores)

# model spec ----
rf_model <- 
  rand_forest(
    min_n = tune(),
    mtry = tune(),
    trees = tune()
  ) %>% 
  set_mode("regression") %>% 
  set_engine("ranger")

# define workflows ----
rf_wflow <- 
  workflow() %>% 
  add_model(rf_model) %>%
  add_recipe(recipe_rf)

# hyperparameter tuning values ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(c(19, 21))) |> 
  update(min_n = min_n(c(1, 3))) |> 
  update(trees = trees(c(500, 2500)))

rf_grid <- grid_regular(rf_params, levels = c(3, 3, 5))

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("RF Model") # start clock

# tuning code in here
tune_rf_fe <- tune_grid(
  rf_wflow, 
  resamples = rf_folds,
  grid = rf_grid,
  metrics = metric_set(mae),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_rf_fe, file = here("results/tune_rf_fe.rda"))
save(tictoc_rf_fe, file = here("results/tictoc_rf_fe.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_rf_fe |> 
  extract_workflow(tune_rf_fe) |>  
  finalize_workflow(select_best(tune_rf_fe, metric = "mae"))

# train final model 
# set seed
set.seed(1234)
final_fit_rf_fe <- fit(final_wflow, train_test)
metrics_rf_fe <- bind_cols(train_test_rf, predict(final_fit_rf_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_rf_fe, file = here("results/metrics/metrics_rf_fe.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_rf <- bind_cols(test, predict(final_fit_rf_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted) 

# save submission test ----
write_csv(submission_test_rf, file = here("submissions/submission_test_rf.csv"))
