# Regression Prediction Problem ----
# Define and fit elastic net model on fe recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
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

# model spec ----
en_spec <- 
  linear_reg(
    penalty = tune(),
    mixture = tune()
  ) |> 
  set_mode("regression") |> 
  set_engine("glmnet")

# define workflow
en_wflow <-
  workflow() |> 
  add_model(en_spec) |> 
  add_recipe(recipe_fe)

# fit workflows/models to folded data----
keep_wflow <- control_resamples(save_workflow = TRUE)

# hyperparameter tuning values ----
en_params <- hardhat::extract_parameter_set_dials(en_spec) |> 
  update(mixture = mixture(c(0,1)),
         penalty = penalty(c(-5,0)))

en_grid <- grid_regular(en_params, levels = 5)

# fit workflows/models ----
set.seed(1234)
tic.clearlog() # clear log
tic("en with fe") # start clock

tune_en_fe <-
  en_wflow |> 
  tune_grid(
    folds,
    grid = en_grid, 
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# extract runtime
time_log <- tic.log(format = FALSE)

tictoc_en_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_en_fe, file = here("results/tune_en_fe.rda"))
save(tictoc_en_fe, file = here("results/tictoc_en_fe.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_en_fe |> 
  extract_workflow(tune_en_fe) |>  
  finalize_workflow(select_best(tune_en_fe, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_en_fe <- fit(final_wflow, train)

metrics_en_fe <- bind_cols(train_test, predict(final_tune_en_fe, train_test)) |>
  select(price_log10, .pred) |>
  mutate(price = 10^price_log10) |>
  mutate(preds = 10^.pred) |>
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_en_fe, file = here("results/metrics/metrics_en_fe.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_en <- bind_cols(test, predict(final_tune_en_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted) 

# save submission test ----
write_csv(submission_test_en, file = here("submissions/submission_test_en.csv"))

