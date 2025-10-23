### Regression Prediction Problem ----
# Analysis of trained models to determine submission models ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))
load(here("data/test.rda"))
load(here("data/train_test.rda"))

###############################################################################
# Model metrics ----
###############################################################################

# loading all model metrics tables ----
load(here("results/comb_metrics_fe.rda"))
load(here("results/comb_metrics_ks.rda"))
load(here("results/metrics/metrics_ensemble.rda"))

# make combine metric table ----
table_metrics_ensemble <- metrics_ensemble |> 
  rename(MAE = .estimate) |> 
  mutate(model = "Boosted Ensemble Model",
         recipe = "Feature Engineering", 
         runtime = "27.094")
  
all_metrics <- comb_metrics_fe |> full_join(comb_metrics_ks) |> 
  bind_rows(table_metrics_ensemble) |> 
  select(-.metric, -.estimator) |> 
  arrange(MAE)

table_all_metrics <- all_metrics |> 
  kbl() |> 
  kable_styling()

# save 
save(all_metrics, file = here("results/all_metrics.rda"))
save(table_all_metrics, file = here("figures_tables/table_all_metrics.rda"))

###############################################################################
# Parameters of final selected models ----
###############################################################################

# best parameters for BT LGBM ----
load(here("results/tune_lgbm_bt_fe.rda"))

parameters_lbt <- autoplot(tune_lgbm_bt_fe) 
table_lbt_parameters <- show_best(tune_lgbm_bt_fe, metric = "mae") |> slice_head(n = 1) |> 
  select(-.estimator, -n, -.metric, -mean, -std_err) |> 
  kbl() |> kable_styling()

save(parameters_lbt, file = here("figures_tables/parameters_lbt.rda"))
save(table_lbt_parameters, file = here("figures_tables/table_lbt_parameters.rda"))


# best parameters for RF ----
load(here("results/tune_rf_fe.rda"))

parameters_rf <- autoplot(tune_rf_fe)
table_rf_parameters <- show_best(tune_rf_fe, metric = "mae") |> slice_head(n = 1) |> 
  select(-.estimator, -n, -.metric, -mean, -std_err) |> 
  kbl() |> kable_styling()

save(parameters_rf, file = here("figures_tables/parameters_rf.rda"))
save(table_rf_parameters, file = here("figures_tables/table_rf_parameters.rda"))

###############################################################################
# Residuals of final selected models ----
###############################################################################

# lgbm bt residuals ----
load(here("results/tune_lgbm_bt_fe.rda"))

final_wflow <- tune_lgbm_bt_fe |> 
  extract_workflow(tune_lgbm_bt_fe) |>  
  finalize_workflow(select_best(tune_lgbm_bt_fe, metric = "mae"))

set.seed(1234)
final_fit_lgbm_bt_fe <- fit(final_wflow, train_test)

lgbm_bt_residuals <- bind_cols(train_test, predict(final_fit_lgbm_bt_fe, train_test)) |> 
  select(price_log10, .pred) |>
  mutate(price = 10^price_log10) |>
  mutate(preds = 10^.pred) |>
  mutate(residual = price - preds)

avg_residuals_lgbm_bt <- mean(lgbm_bt_residuals$residual)

mean(rf_residuals$preds)

10^2.065876

lm(residual ~ preds, data = lgbm_bt_residuals)

lgbm_bt_residuals_plot <- ggplot(lgbm_bt_residuals, mapping = aes(x = preds, y = residual)) +
  geom_point() + 
  geom_abline(slope = 0.05, linetype = 3) +
  theme_minimal() + 
  xlim(0, 1000) + 
  ylim(0, 75) + 
  labs(title = "Predicted vs. Actual Prices (USD, Original Scale) for Light GBM Boosted Tree Model",
       subtitle = "Average residual is 9.08",
       caption = "Note: removed extreme outliers for visualization purposes",
       x = "Predicted Prices (USD)",
       y = "Actual Prices (USD)")

save(lgbm_bt_residuals_plot, file = here("figures_tables/lgbm_bt_residuals_plot.rda"))

# rf residuals ----
load(here("results/tune_rf_fe.rda"))

final_wflow <- tune_rf_fe |> 
  extract_workflow(tune_rf_fe) |>  
  finalize_workflow(select_best(tune_rf_fe, metric = "mae"))

set.seed(1234)
final_fit_rf_fe <- fit(final_wflow, train_test)

rf_residuals <- bind_cols(train_test, predict(final_fit_rf_fe, train_test)) |> 
  select(price_log10, .pred) |>
  mutate(price = 10^price_log10) |>
  mutate(preds = 10^.pred) |>
  mutate(residual = price - preds)

avg_residuals_rf <- mean(rf_residuals$residual)

lm(residual ~ preds, data = rf_residuals)

rf_residuals_plot <- ggplot(rf_residuals, mapping = aes(x = preds, y = residual)) +
  geom_point() + 
  geom_abline(slope = 0.3, linetype = 3) +
  theme_minimal() + 
  xlim(0, 1000) + 
  ylim(0, 1000) + 
  labs(title = "Predicted vs. Actual Prices (USD, Original Scale) for Random Forest Model",
       subtitle = "Average residual is 83.90",
       caption = "Note: removed extreme outliers for visualization purposes",
       x = "Predicted Prices (USD)",
       y = "Actual Prices (USD)")

save(rf_residuals_plot, file = here("figures_tables/rf_residuals_plot.rda"))

