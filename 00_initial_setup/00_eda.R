## Regression Prediction Problem ----
## Exploratory Data Analysis ----

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts 
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))
load(here("data/train_regression.rda"))

# check missingness 
train |> 
  skimr::skim_without_charts(price_log10) 

train |> 
  skimr::skim_without_charts(price_log10)

# inspect target variable (price_log10) ----
price_density <- train_regression |> 
  ggplot(aes(price)) + 
  geom_density() + 
  labs(title = "Distribution of Price (Original Scale)") + 
  theme_minimal()

price10_density <- train |> 
  ggplot(aes(price_log10)) + 
  geom_density() + 
  labs(title = "Distribution of Price (Log10)") + 
  theme_minimal()

# save ----
save(price10_density, file = here("figures_tables/price10_density.rda"))
save(price_density, file = here("figures_tables/price_density.rda"))

