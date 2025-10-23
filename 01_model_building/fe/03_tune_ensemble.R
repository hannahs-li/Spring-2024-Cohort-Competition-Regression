# Regression Prediction Problem ----
# Train & explore ensemble model ----

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(tictoc)
library(bonsai)

# Handle common conflicts
tidymodels_prefer()

# load data ----
load(here("data/train_train.rda"))
load(here("data/train_test.rda"))

# Load candidate model info ----
load(here("results/comb_metrics_fe.rda"))

# top candidates are both bt models 
load(here("results/tune_lgbm_bt_fe.rda"))
load(here("results/tune_bt_fe.rda"))

# Build ensemble model ----
tic.clearlog() # clear log
tic("Ensemble Model") # start clock

# create data stack
data_stack <- 
  stacks() |> 
  add_candidates(tune_lgbm_bt_fe) |> 
  add_candidates(tune_bt_fe)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
# blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions (tuning step, set seed)
set.seed(1234)
stack_blend <-
  data_stack |> 
  blend_predictions()

# Look at weights 
bt_wt <- stack_blend |> 
  collect_parameters("tune_bt_fe") |> 
  filter(coef != 0) |> 
  select(member, coef)

bt_lgbm_wt <- stack_blend |> 
  collect_parameters("tune_lgbm_bt_fe") |> 
  filter(coef != 0) |> 
  select(member, coef)

table_wt <- bt_wt |> 
  bind_rows(bt_lgbm_wt) |> 
  arrange(-coef) |> 
  rename(model = member, 
         weight = coef)

# save blended model stack for reproducibility & easy reference (for report) and wt table 
save(stack_blend, file = here("results/stack_blend.rda"))
save(table_wt, file = here("figures_tables/table_wt.rda"))

# fit to training set ----
model_stack <-
  stack_blend |>
  fit_members(control_grid(save_workflow = TRUE))

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_ensemble <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# save ---- 
save(model_stack, file = here("results/fit_ensemble.rda"))
save(tictoc_ensemble, file = here("results/tictoc_ensemble.rda"))

# Explore the blended model stack ----
autoplot_stack <- autoplot(stack_blend, metric = "mae") + theme_minimal()
# show relationship more directly 
autoplot_member <- autoplot(stack_blend, type = "members") + theme_minimal()
# see top results 
autoplot_weight <- autoplot(stack_blend, type = "weights") + theme_minimal()

# save 
save(autoplot_stack, file = here("figures_tables/autoplot_stack.rda"))
save(autoplot_member, file = here("figures_tables/autoplot_member.rda"))
save(autoplot_weight, file = here("figures_tables/autoplot_weight.rda"))

# collect metrics ----
metrics_ensemble <- bind_cols(train_test, predict(model_stack, train_test)) |> 
  select(price_log10, .pred) |> 
  mutate(price = price_log10) |> 
  mutate(preds = .pred) |> 
  mae(truth = price, estimate = preds)

save(metrics_ensemble, file = here("results/metrics/metrics_ensemble.rda"))

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_ensemble <- bind_cols(test, predict(model_stack, test)) |>
   mutate(predicted = 10^.pred) |>
   select(id, predicted)

# save submission test ----
write_csv(submission_test_ensemble, file = here("submissions/submission_test_ensemble.csv"))

