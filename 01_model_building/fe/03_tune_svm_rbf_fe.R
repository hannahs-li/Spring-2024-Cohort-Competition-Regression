# Regression Prediction Problem ----
# Define and fit radial basis function vector machine with fe recipe ---

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
load(here("recipes/recipe_fe.rda"))

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
  add_recipe(recipe_fe)

# hyperparameter tuning values ----
svm_rbf_param <- hardhat::extract_parameter_set_dials(svm_rbf_model)
svm_rbf_grid <- grid_latin_hypercube(svm_rbf_param, size = 53)

# fit workflow/model ----
tic("SVM RBF: FE") # start clock

# tuning code in here
tune_svm_rbf_fe <- svm_rbf_wflow |> 
  tune_grid(
    resamples = folds, 
    grid = svm_rbf_grid,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_rbf_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

stopCluster(cl)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_rbf_fe, file = here("results/tune_svm_rbf_fe.rda"))
save(tictoc_svm_rbf_fe, file = here("results/tictoc_svm_rbf_fe.rda"))

# collect metrics ----
# extract final workflow 
final_wflow <- tune_svm_rbf_fe |> 
  extract_workflow(tune_svm_rbf_fe) |>  
  finalize_workflow(select_best(tune_svm_rbf_fe, metric = "mae"))

# train final model 
set.seed(1234)
final_tune_svm_rbf_fe <- fit(final_wflow, train_train)

metrics_svm_rbf_fe <- bind_cols(train_test, predict(final_tune_svm_rbf_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_svm_rbf_fe, file = here("results/metrics/metrics_svm_rbf_fe.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_svm_rbf <- bind_cols(test, predict(final_tune_svm_rbf_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted) 

# save submission test
write_csv(submission_test_svm_rbf, file = here("submissions/submission_test_svm_rbf.csv"))

