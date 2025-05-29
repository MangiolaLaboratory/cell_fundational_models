library("zellkonverter")
library("AnnotationDbi")
library("org.Hs.eg.db")

#file <- system.file("anndata", "/Users/gp/Desktop/Pan_Cancer_tumor_infiltrating_tcells/data 6/velo/adata_CD4.h5ad", package = "zellkonverter")

data <- zellkonverter::readH5AD("/Users/gp/Desktop/Pan_Cancer_tumor_infiltrating_tcells/data 6/velo/adata_CD8.h5ad")

#gene symbols
gene_symbols <- rownames(data)

# Convert to Ensembl IDs
ensembl_ids <- AnnotationDbi::mapIds(
  org.Hs.eg.db::org.Hs.eg.db,
  keys = gene_symbols,
  column = "ENSEMBL",
  keytype = "SYMBOL",
  multiVals = "first"
)

# Add Ensembl IDs to rowData
rowData(data)$ensembl_id <- ensembl_ids

zellkonverter::writeH5AD(data, "/Users/gp/Desktop/cd8_ensembl_ids.h5ad")
