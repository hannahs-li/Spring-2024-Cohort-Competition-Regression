### Regression Prediction Problem ----
# Analysis of trained models ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))
load(here("data/test.rda"))

# based all submissions on model with lowest MAE so far 
###############################################################################
# Submission 1 (lm) ----
###############################################################################
# train final model 
# set seed
set.seed(1)
final_lm_fit <- fit(lm_wflow, train)

# metric_set <- metric_set(rmse)

# Final model trained metrics 
submission_lm <- bind_cols(test, predict(final_lm_fit, test)) |> 
  mutate(predicted = .pred) |> 
  select(id, predicted)

submission_lm[is.na(submission_lm)] <- 0

# turn to csv
write_csv(submission_lm, file = here("submissions/submission_lm.csv"))

###############################################################################
# Submission 2 (bt) ----
###############################################################################
# load model 
load(here("results/bt_tune_fe.rda"))

# train final model 
final_wflow_bt <- bt_tune_fe |> 
  extract_workflow() |> 
  finalize_workflow(select_best(bt_tune_fe, metric = "rmse"))

# set seed
set.seed(1)
final_fit_bt <- fit(final_wflow_bt, train)

# Final model trained metrics 
submission_bt <- bind_cols(test, predict(final_fit_bt, test)) |> 
  mutate(predicted = .pred) |> 
  select(id, predicted)

# submission_lm[is.na(submission_lm)] <- 0

# turn to csv
write_csv(submission_bt, file = here("submissions/submission_lm.csv"))

###############################################################################
# Submission (bt xgboost) ----
###############################################################################
load(here("data/test.rda"))
load(here("results/tune_bt_fe.rda"))

final_wflow <- tune_bt_fe |> 
  extract_workflow(tune_bt_fe) |>  
  finalize_workflow(select_best(tune_bt_fe, metric = "mae"))

set.seed(1234)
final_fit_bt_fe <- fit(final_wflow, train)

submission_test_bt <- bind_cols(test, predict(final_fit_bt_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted)

write_csv(submission_test_bt, file = here("submissions/submission_test_bt.csv"))

###############################################################################
# Submission (bt lgbm) ----
###############################################################################
load(here("data/test.rda"))
load(here("results/tune_lgbm_bt_fe.rda"))

final_wflow <- tune_lgbm_bt_fe |> 
  extract_workflow(tune_lgbm_bt_fe) |>  
  finalize_workflow(select_best(tune_lgbm_bt_fe, metric = "mae"))

set.seed(1234)
final_tune_bt_lgbm_fe <- fit(final_wflow, train)

submission_test_bt_lgbm <- bind_cols(test, predict(final_tune_bt_lgbm_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted)

write_csv(submission_test_bt_lgbm, file = here("submissions/submission_test_bt_lgbm.csv"))
