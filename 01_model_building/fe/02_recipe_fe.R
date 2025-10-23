# Regression ----
# Setup fe recipe 

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data ----
load(here("data/train.rda"))

# fe recipe ----
# build recipe 
recipe_fe <- recipe(price_log10 ~., data = train) |> 
  step_rm(id) |> 
  step_novel(neighbourhood_cleansed) |> 
  step_novel(host_neighbourhood) |> 
  step_novel(host_location) |> 
  step_interact(~first_review:last_review) |> 
  step_interact(~review_scores_rating:
                  reviews_per_month:
                  review_scores_accuracy: 
                  review_scores_cleanliness: 
                  review_scores_checkin: 
                  review_scores_communication: 
                  review_scores_location: 
                  review_scores_value
                ) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors()) |>
  step_other(threshold = 0.05) |>
  step_dummy(all_nominal_predictors()) |>
  step_nzv(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors())

# check recipe 
recipe_fe |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

# rf fe recipe ----
recipe_rf <- recipe(price_log10 ~., data = train) |> 
  step_rm(id) |> 
  step_interact(~first_review:last_review) |> 
  step_interact(~review_scores_rating:
                  reviews_per_month:
                  review_scores_accuracy: 
                  review_scores_cleanliness: 
                  review_scores_checkin: 
                  review_scores_communication: 
                  review_scores_location: 
                  review_scores_value
  ) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = 0.05) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_nzv(all_predictors()) |>
  step_normalize(all_numeric_predictors()) 

recipe_rf |>
  prep() |>
  bake(new_data = NULL)

# save recipe ----
save(recipe_fe, file = here("recipes/recipe_fe.rda"))
save(recipe_rf, file = here("recipes/recipe_rf.rda"))