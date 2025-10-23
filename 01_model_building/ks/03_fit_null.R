# Regression Prediction Problem ----
# Define and fit null model or baseline ----

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

# model specifications ----
null_spec <- 
  null_model() |> 
  set_engine("parsnip") |> 
  set_mode("regression") 

# define workflows ----
null_wflow <- workflow() |> 
  add_model(null_spec) |> 
  add_recipe(recipe_ks)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Null") # start clock

set.seed(1234)
fit_null <- 
  null_wflow |> 
  fit_resamples(
    resamples = folds, 
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_null <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_null, file = here("results/fit_null.rda"))
save(tictoc_null, file = here("results/tictoc_null.rda"))

# collect metrics ----
# train final model 
set.seed(1234)
final_fit_null <- fit(null_wflow, train_train)

metrics_null <- bind_cols(train_test, predict(final_fit_null, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_null, file = here("results/metrics/metrics_null.rda"))
