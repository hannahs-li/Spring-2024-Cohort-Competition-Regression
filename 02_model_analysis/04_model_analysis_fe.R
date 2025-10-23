# Regression Prediction Problem ----
# Analysis of trained models ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load fits/tunings for all fe models ----
list.files(
  here("results/"),
  "_fe",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)

# load metrics for all fe models ----
list.files(
  here("results/metrics"),
  "_fe",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)

# model name here |> show_best()
comb_metrics_fe <- metrics_bt_fe |> 
  mutate(model = "Boosted Tree (XGBoost)",
         recipe = "Feature Engineering",
         runtime = "1872.672") |> 
  bind_rows(metrics_en_fe |> 
        mutate(model = "Elastic Net",
               recipe = "Feature Engineering", 
               runtime = "91.785")
  ) |> 
  bind_rows(metrics_lm_fe |> 
              mutate(model = "Linear Regression Model",
                     recipe = "Feature Engineering", 
                     runtime = "4.306")
  ) |>
  bind_rows(metrics_knn_fe |> 
              mutate(model = "K-Nearest Neighbor",
                     recipe = "Feature Engineering", 
                     runtime = "2.684")
  ) |>
  bind_rows(metrics_mars_fe |> 
              mutate(model = "Multivariate Adaptive Regression Spline (MARS)",
                     recipe = "Feature Engineering", 
                     runtime = "27.06")
  ) |>
  bind_rows(metrics_mlp_fe |> 
              mutate(model = "Multilayer Perceptron (MLP)",
                     recipe = "Feature Engineering", 
                     runtime = "65.931")
  ) |>
  bind_rows(metrics_rf_fe |> 
              mutate(model = "Random Forest",
                     recipe = "Feature Engineering", 
                     runtime = "2086.902")
  ) |>
  bind_rows(metrics_svm_poly_fe |> 
              mutate(model = "Polynomial Support Vector Machine (SVM Poly)",
                     recipe = "Feature Engineering", 
                     runtime = "50.283")
  ) |> 
  bind_rows(metrics_svm_rbf_fe |> 
              mutate(model = "Radial Basis Function Support Vector Machine (SVM RBF)",
                     recipe = "Feature Engineering", 
                     runtime = "193.639")
  ) |> 
  bind_rows(metrics_bt_lgbm_fe |> 
              mutate(model = "Boosted Tree (LGBM)",
                     recipe = "Feature Engineering", 
                     runtime = "233.217")
  ) |> 
  mutate(MAE = .estimate) |> 
  select(model, recipe, MAE, runtime) |> 
  arrange(MAE) 

table_comb_metrics_fe <- comb_metrics_fe |> 
  kbl() |> 
  kable_styling()


# # plotting model results
# en_plot <- autoplot(en_tuned, metric = "rmse", select_best = TRUE, std_errs = 1) + 
#   theme_minimal() + 
#   labs(
#     y = "RMSE",
#     title = "Elastic net model penalty and mixture vs. RMSE",
#     subtitle = "Using the tuned elastic net model",
#     color = "Penalty"
#   )
# 
# 
# save(en_plot, file = here("figures_tables/autoplots/en_plot.rda"))
# save(en_best_param, file = here("figures_tables/tables/en_best_param.rda"))


# saving out results ----
save(comb_metrics_fe, file = here("results/comb_metrics_fe.rda"))
save(table_comb_metrics_fe, file = here("figures_tables/table_comb_metrics_fe.rda"))
