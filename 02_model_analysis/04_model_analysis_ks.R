# Regression Prediction Problem ----
# Analysis of trained models ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load fits/tunings for all ks models ----
list.files(
  here("results/"),
  "_ks",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)

# load metrics for all ks models ----
list.files(
  here("results/metrics"),
  "_ks",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)
load(here("results/metrics/metrics_null.rda"))

# model name here |> show_best()
comb_metrics_ks <- metrics_bt_ks |> 
  mutate(model = "Boosted Tree (XGBoost)",
         recipe = "Kitchen Sink",
         runtime = "1799.047") |> 
  bind_rows(metrics_bt_lgbm_ks |>
              mutate(model = "Boosted Tree (LGBM)",
                     recipe = "Kitchen Sink",
                     runtime = "222.731")
  ) |>
  bind_rows(metrics_en_ks |> 
        mutate(model = "Elastic Net",
               recipe = "Kitchen Sink", 
               runtime = "219.126")
  ) |> 
  bind_rows(metrics_lm_ks |> 
              mutate(model = "Linear Regression Model",
                     recipe = "Kitchen Sink", 
                     runtime = "3.217")
  ) |>
  bind_rows(metrics_knn_ks |> 
              mutate(model = "K-Nearest Neighbor",
                     recipe = "Kitchen Sink", 
                     runtime = "4.53")
  ) |>
  bind_rows(metrics_mars_ks |> 
              mutate(model = "Multivariate Adaptive Regression Spline (MARS)",
                     recipe = "Kitchen Sink", 
                     runtime = "1055.373")
  ) |>
  bind_rows(metrics_mlp_ks |> 
              mutate(model = "Multilayer Perceptron (MLP)",
                     recipe = "Kitchen Sink", 
                     runtime = "1804.163")
  ) |>
  bind_rows(metrics_null |> 
               mutate(model = "Null",
                      recipe = "Kitchen Sink", 
                      runtime = "2.945")
  ) |>
  bind_rows(metrics_svm_poly_ks |> 
              mutate(model = "Polynomial Support Vector Machine (SVM Poly)",
                     recipe = "Kitchen Sink", 
                     runtime = "1941.898")
  ) |> 
  bind_rows(metrics_svm_rbf_ks |> 
              mutate(model = "Radial Basis Function Support Vector Machine (SVM RBF)",
                     recipe = "Kitchen Sink", 
                     runtime = "131.428")
  ) |> 
  mutate(MAE = .estimate) |> 
  select(model, recipe, MAE, runtime) |> 
  arrange(MAE) 

table_comb_metrics_ks <- comb_metrics_ks |> 
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
save(comb_metrics_ks, file = here("results/comb_metrics_ks.rda"))
save(table_comb_metrics_ks, file = here("figures_tables/table_comb_metrics_ks.rda"))
