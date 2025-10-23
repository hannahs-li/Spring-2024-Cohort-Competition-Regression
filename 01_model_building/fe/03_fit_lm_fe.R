# Regression Prediction Problem ----
# Define and fit ordinary linear model with fe recipe 

## load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(ranger)
library(doMC)
library(tictoc)

# handle common conflicts 
tidymodels_prefer()

# parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load folds and recipe 
load(here("data/folds.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))
load(here("recipes/recipe_fe.rda"))

# fit model -----
# model spec 
lm_spec <- 
  linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression") 

# define workflows
lm_wflow <-  
  workflow() |> 
  add_model(lm_spec) |> 
  add_recipe(recipe_fe)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("LM with FE") # start clock

set.seed(1234) 
fit_lm_fe <- 
  fit_resamples(
    lm_wflow,
    resamples = folds,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_lm_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_lm_fe, file = here("results/fit_lm_fe.rda"))
save(tictoc_lm_fe, file = here("results/tictoc_lm_fe.rda"))

# collect model metrics ----
# extract final wflow 
final_wflow <- fit_lm_fe |> 
  extract_workflow(fit_lm_fe) |>  
  finalize_workflow(select_best(fit_lm_fe, metric = "mae"))

# train final model 
# set seed
set.seed(1234)
final_fit_lm_fe <- fit(final_wflow, train_train)
metrics_lm_fe <- bind_cols(train_test, predict(final_fit_lm_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_lm_fe, file = here("results/metrics/metrics_lm_fe.rda"))

# compare to test submission  
load(here("data/test.rda"))
submission_test_lm <- bind_cols(test, predict(final_fit_lm_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted) 

# save submission test ----
write_csv(submission_test_lm, file = here("submissions/submission_test_lm.csv"))



