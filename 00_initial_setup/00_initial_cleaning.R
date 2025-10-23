## Regression Prediction Problem 
## Data Cleaning, Splitting, and Folding

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts 
tidymodels_prefer()

# load data ----
# clean train 
train_regression <- read_csv(here("data/train_regression.csv"),
                             col_types = cols(id = col_character())) 
  
train_regression$price <- gsub("\\$", "", train_regression$price)
train_regression$price <- gsub("\\,", "", train_regression$price)      

train <- train_regression |> 
  janitor::clean_names() |> 
  mutate(price = as.numeric(price)) |> 
  mutate(price_log10 = log10(price)) |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(across(where(is.logical), as.factor)) |> 
  # mutate(across(where(is.Date), as.numeric)) |> # days since 1/1/1970
  mutate_at(c('host_since', 'first_review', 'last_review'), as.Date, format = '%Y-%m-%d') |> 
  mutate(host_since = as.numeric(format(host_since,'%Y')),
         first_review = as.numeric(format(first_review,'%Y')),
         last_review = as.numeric(format(last_review,'%Y'))) |> 
  mutate(id = as.character(id)) |> 
  mutate(host_response_rate = as.numeric(host_response_rate)) |> 
  mutate(host_acceptance_rate = as.numeric(host_acceptance_rate)) |>
  select(-price) |> 
  mutate(minimum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_nights_avg_ntm = pmin(minimum_maximum_nights, 1125)) 

# convert bathrooms_text to bathrooms 
train <- train |> 
  mutate(bathrooms_text = as.character(bathrooms_text)) |> 
  mutate(bathrooms = gsub("\\bbaths?\\b", "", bathrooms_text)) |> 
  mutate(bathrooms_v2 = gsub("\\bprivate\\b", "", bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bHalf-\\b", "0.5", bathrooms_v2)) |> 
  mutate(shared_bathrooms = grepl('shared', bathrooms_v2)) |> 
  mutate(shared_bathrooms = factor(shared_bathrooms))  |> 
  mutate(bathrooms_v2 = gsub("\\bshared\\b", "", bathrooms_v2)) |> 
  mutate(true_bathrooms = as.numeric(bathrooms_v2)) |> 
  select(-c(bathrooms_text, bathrooms_v2, bathrooms)) 

# clean test 
test_regression <- read_csv(here("data/test_regression.csv"),
                             col_types = cols(id = col_character()))

test <- test_regression |> 
  janitor::clean_names() |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(across(where(is.logical), as.factor)) |> 
  # mutate(across(where(is.Date), as.numeric)) |> # days since 1/1/1970
  mutate_at(c('host_since', 'first_review', 'last_review'), as.Date, format = '%Y-%m-%d') |> 
  mutate(host_since = as.numeric(format(host_since,'%Y')),
         first_review = as.numeric(format(first_review,'%Y')),
         last_review = as.numeric(format(last_review,'%Y'))) |> 
  mutate(id = as.character(id)) |> 
  mutate(host_response_rate = as.numeric(host_response_rate)) |> 
  mutate(host_acceptance_rate = as.numeric(host_acceptance_rate)) |> 
  mutate(minimum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_nights_avg_ntm = pmin(minimum_maximum_nights, 1125))

# convert bathrooms_text to bathrooms 
test <- test |> 
  mutate(bathrooms_text = as.character(bathrooms_text)) |> 
  mutate(bathrooms = gsub("\\bbaths?\\b", "", bathrooms_text)) |> 
  mutate(bathrooms_v2 = gsub("\\bprivate\\b", "", bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bHalf-\\b", "0.5", bathrooms_v2)) |> 
  mutate(shared_bathrooms = grepl('shared', bathrooms_v2)) |> 
  mutate(shared_bathrooms = grepl('Shared', bathrooms_v2)) |> 
  mutate(bathrooms_v2 = gsub("\\bShared half-\\b", "0.5", bathrooms_v2)) |> 
  mutate(shared_bathrooms = factor(shared_bathrooms))  |> 
  mutate(bathrooms_v2 = gsub("\\bshared\\b", "", bathrooms_v2)) |> 
  mutate(true_bathrooms = as.numeric(bathrooms_v2)) |> 
  select(-c(bathrooms_text, bathrooms_v2, bathrooms))


# save dataset ----
save(train, file = here("data/train.rda"))
save(test, file = here("data/test.rda"))
