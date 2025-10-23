# Regression Prediction Problem ----
# Define and fit elastic net model on ks recipe ----

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
  add_recipe(recipe_ks)

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
tic("en with ks") # start clock

tune_en_ks <-
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

tictoc_en_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_en_ks, file = here("results/tune_en_ks.rda"))
save(tictoc_en_ks, file = here("results/tictoc_en_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_en_ks |> 
  extract_workflow(tune_en_ks) |>  
  finalize_workflow(select_best(tune_en_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_en_ks <- fit(final_wflow, train_train)

metrics_en_ks <- bind_cols(train_test, predict(final_tune_en_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_en_ks, file = here("results/metrics/metrics_en_ks.rda"))



