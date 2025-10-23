# Regression Prediction Problem ----
# Define and fit radial basis function vector machine with ks recipe ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(tictoc)
library(doMC)

# Handle conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load training data
load(here("data/folds.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks.rda"))

# model specifications ----
svm_rbf_model <- svm_rbf(
  mode = "regression",
  cost = tune(),
  rbf_sigma = tune()
) |> 
  set_engine("kernlab")

# define workflows ----
svm_rbf_wflow <- workflow() |> 
  add_model(svm_rbf_model) |> 
  add_recipe(recipe_ks)

# hyperparameter tuning values ----
svm_rbf_param <- hardhat::extract_parameter_set_dials(svm_rbf_model)
svm_rbf_grid <- grid_latin_hypercube(svm_rbf_param, size = 60)

# fit workflow/model ----
tic("SVM RBF: KS") # start clock

# tuning code in here
tune_svm_rbf_ks <- svm_rbf_wflow |> 
  tune_grid(
    resamples = folds, 
    grid = svm_rbf_grid,
    control = control_grid(save_workflow = TRUE),
    metrics = metric_set(mae)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_rbf_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

stopCluster(cl)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_rbf_ks, file = here("results/tune_svm_rbf_ks.rda"))
save(tictoc_svm_rbf_ks, file = here("results/tictoc_svm_rbf_ks.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_svm_rbf_ks |> 
  extract_workflow(tune_svm_rbf_ks) |>  
  finalize_workflow(select_best(tune_svm_rbf_ks, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_svm_rbf_ks <- fit(final_wflow, train_train)

metrics_svm_rbf_ks <- bind_cols(train_test, predict(final_tune_svm_rbf_ks, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_svm_rbf_ks, file = here("results/metrics/metrics_svm_rbf_ks.rda"))
