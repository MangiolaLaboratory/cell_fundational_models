# -*- coding: utf-8 -*-
"""getdata.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Wl5WGu0QUvUD5baq-r-AIHa1yj_HmYTw
"""

!pip install cellxgene-census

import cellxgene_census
help (cellxgene_census)

with cellxgene_census.open_soma() as census:
    adata = cellxgene_census.get_anndata(
        census = census,
        organism = "Homo sapiens",
        var_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']",
        obs_value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = {"obs": ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease","ethnicity","age"]},
    )

    print(adata)

with cellxgene_census.open_soma() as census:

    # Reads SOMADataFrame as a slice
    cell_metadata = census["census_data"]["homo_sapiens"].obs.read(
        value_filter = "sex == 'female' and cell_type in ['microglial cell', 'neuron']",
        column_names = ["assay", "cell_type", "tissue", "tissue_general", "suspension_type", "disease"]
    )

    # Concatenates results to pyarrow.Table
    cell_metadata = cell_metadata.concat()

    # Converts to pandas.DataFrame
    cell_metadata = cell_metadata.to_pandas()

    print(cell_metadata)

import anndata as ad
import pandas as pd

# Assuming you have a DataFrame called 'df'
adata_c = ad.AnnData(cell_metadata)
adata_c.write_h5ad("my_data.h5ad")

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .