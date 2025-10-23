# Regression Prediction Problem ----
# Define and fit knn model on fe recipe

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# parallel processing 
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores)

# load training data
load(here("data/folds.rda"))
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_fe.rda"))

# model specifications ----
knn_spec <- 
  nearest_neighbor(neighbors = 20) |>
  set_mode("regression") |> 
  set_engine("kknn") 

# define workflows ----
knn_wflow <- 
  workflow() |> 
  add_model(knn_spec) |> 
  add_recipe(recipe_fe)

tic.clearlog() # clear log
tic("knn with fe") # start clock

# fit workflows/models to folded data----
set.seed(12345)
fit_knn_fe <- knn_wflow |> 
  fit_resamples(
    resamples = folds,
    metrics = metric_set(mae),
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE)

# extract run time 
time_log <- tic.log(format = FALSE)

tictoc_knn_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_knn_fe, file = here("results/fit_knn_fe.rda"))
save(tictoc_knn_fe, file = here("results/tictoc_knn_fe.rda"))


# collect metrics ----
# extract final workflow 
final_wflow <- fit_knn_fe |> 
  extract_workflow(fit_knn_fe) |>  
  finalize_workflow(select_best(fit_knn_fe, metric = "mae"))

# train final model 
set.seed(1234)
final_fit_knn_fe <- fit(final_wflow, train_train)

metrics_knn_fe <- bind_cols(train_test, predict(final_fit_knn_fe, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = 10^price_log10) |> 
  mutate(preds = 10^.pred) |> 
  mae(truth = price, estimate = preds)

# save model metrics ----
save(metrics_knn_fe, file = here("results/metrics/metrics_knn_fe.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_knn <- bind_cols(test, predict(final_fit_knn_fe, test)) |> 
  mutate(predicted = .pred) |> 
  mutate(predicted = 10^predicted) |> 
  select(id, predicted) 

# save submission test ----
write_csv(submission_test_knn, file = here("submissions/submission_test_knn.csv"))
