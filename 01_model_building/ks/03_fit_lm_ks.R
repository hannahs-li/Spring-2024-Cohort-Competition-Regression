# Regression Prediction Problem ----
# Define and fit ordinary linear model with ks recipe ----

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
  add_recipe(recipe_ks)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("LM with KS") # start clock

set.seed(1234) 
fit_lm_ks <- 
  fit_resamples(
    lm_wflow,
    resamples = folds,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_lm_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_lm_ks, file = here("results/fit_lm_ks.rda"))
save(tictoc_lm_ks, file = here("results/tictoc_lm_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- fit_lm_ks |> 
  extract_workflow(fit_lm_ks) |>  
  finalize_workflow(select_best(fit_lm_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_fit_lm_ks <- fit(final_wflow, train_train)

metrics_lm_ks <- bind_cols(train_test, predict(final_fit_lm_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_lm_ks, file = here("results/metrics/metrics_lm_ks.rda"))