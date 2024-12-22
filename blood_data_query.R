library(cellNexus)
library(dplyr)
library(tidyr)

samples_per_decade = 10

sce = 
  get_metadata() |> 
  filter(!file_id_cellNexus_single_cell |> is.na()) |> 
  filter(tissue_groups == "blood") |> 
  filter(assay |> str_detect("10x")) |> 
  filter(disease=="normal") |> 
  filter(age_days > 365) |> 
  mutate(age_decades = ceiling(age_days / 365 / 10)) |> 
  as_tibble() |> 
  nest(data = -c(sample_id, age_decades)) |> 
  with_groups(age_decades, slice, seq_len(samples_per_decade)) |> 
  unnest(data) |> 
  
  get_single_cell_experiment(atlas_name = "cellxgene")
