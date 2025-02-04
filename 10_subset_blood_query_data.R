library(cellNexus)
library(dplyr)
library(tidyr)
library(purrr)
library(SummarizedExperiment)

my_samples <-
  get_metadata() |>
  
  # Filter
  filter(!is.na(file_id_cellNexus_single_cell),
         tissue_groups == "blood",
         str_detect(assay, "10x"),
         disease == "normal") |>
  count(sample_id, sex, dataset_id) |>
  
  # Filter bis samples
  filter(n>2000) |>
  select(-n) |>
  as_tibble() |>
  
  # Filter big datasets
  nest(sample_ids = sample_id) |>
  mutate(n_sample = map_int(sample_ids, nrow)) |>
  filter( n_sample >=10) |>
  
  # Select only 10 samples per sex
  mutate(sample_ids = map(sample_ids, head, 10)) |>
  
  # changing 

  # Filter dataset with both sexes with > 10 samples
  add_count(dataset_id) |>
  filter(n == 2) |>
  
  # Get sample_id
  unnest(sample_ids) |>
  pull(sample_id)


# Get 10 cells
cells <-
  get_metadata() |>
  filter(sample_id %in% my_samples) |>
  mutate(published_at = published_at |> as.character(),
            revised_at = revised_at |> as.character()) |>
  select(-run_from_cell_id,-x_approximate_distribution) |>
  get_single_cell_experiment()
cells |> writeH5AD()

# remove run_from_cell_id and x_aproximate_distribution and change published at and revised character to char to remove NaN
cells <- cells |> mutate(published_at = published_at |> as.character(),
                         revised_at = revised_at |> as.character()) |>
  select(-run_from_cell_id,-x_approximate_distribution)
zellkonverter::writeH5AD(cells, "dataset_20_20.h5ad")


