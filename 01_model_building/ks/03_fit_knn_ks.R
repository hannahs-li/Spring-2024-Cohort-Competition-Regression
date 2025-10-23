# Regression Prediction Problem ----
# Define and fit knn model on ks recipe ----

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
knn_spec <- 
  nearest_neighbor(neighbors = 20) |>
  set_mode("regression") |> 
  set_engine("kknn") 

# define workflows ----
knn_wflow <- 
  workflow() |> 
  add_model(knn_spec) |> 
  add_recipe(recipe_ks)

tic.clearlog() # clear log
tic("knn with ks") # start clock

# fit workflows/models to folded data----
set.seed(12345)
fit_knn_ks <- knn_wflow |> 
  fit_resamples(
    resamples = folds,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# extract run time 
time_log <- tic.log(format = FALSE)

tictoc_knn_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_knn_ks, file = here("results/fit_knn_ks.rda"))
save(tictoc_knn_ks, file = here("results/tictoc_knn_ks.rda"))


# collect metrics ----
# extract final workflow 
final_wflow <- fit_knn_ks |> 
  extract_workflow(fit_knn_ks) |>  
  finalize_workflow(select_best(fit_knn_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_fit_knn_ks <- fit(final_wflow, train_train)

metrics_knn_ks <- bind_cols(train_test, predict(final_fit_knn_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_knn_ks, file = here("results/metrics/metrics_knn_ks.rda"))
