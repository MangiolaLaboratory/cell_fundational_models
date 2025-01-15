library(cellNexus)
library(dplyr)
library(tidyr)
library(stringr)
library(zellkonverter)
library(glue)
library(purrr)
library(SummarizedExperiment)
library(SingleCellExperiment)

samples_per_decade = 10
number_of_cells_in_batch = 100

processed_data <- get_metadata() |>
  filter(!file_id_cellNexus_single_cell |> is.na()) |>
  filter(tissue_groups == "blood") |>
  filter(assay |> str_detect("10x")) |>
  filter(disease=="normal") |>
  filter(age_days > 365) |>
  mutate(age_decades = ceiling(age_days / 365 / 10))

batched_data <- processed_data |>
  mutate(batch_id = (row_number() - 1) %/% number_of_cells_in_batch) |>
  as_tibble()

first_batch <- batched_data |>
  filter(batch_id == 0) |>
  as_tibble()

sce <- first_batch |>
  get_single_cell_experiment(atlas_name = "cellxgene")

cd <- colData(sce)
cd[] <- lapply(cd, function(x) {
  x <- ifelse(is.na(x) | is.null(x), "", as.character(x))
  x <- gsub("[[:cntrl:]]", "", x)
  return(x)
})
colData(sce) <- DataFrame(cd)
zellkonverter::writeH5AD(sce, glue("batch_0.h5ad"))
