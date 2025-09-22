# %%
#Code to change cuda version to 12.2.2
import os
os.environ["PATH"] += os.pathsep + "/gpfs/apps/icl/software/CUDA/12.2.2/nvvm/bin:/gpfs/apps/icl/software/CUDA/12.2.2/bin"
LD_LIBRARY_PATH="/apps/icl/software/CUDAcompat/12.2-535.161.08/lib"
os.environ["LD_LIBRARY_PATH"]=os.pathsep+LD_LIBRARY_PATH

# %% [markdown]
# # Notebook to create a pipeline to analyse the gene to gene attention of cell types for Geneformer finetuned to classify sex of the patient from rna counts.
# 
# ## workflow : Produce an average attention matrix for each cell type (average of cells belonging to that celltype) and the genes (token_ids). Plot the attention scores in a heatmap for each celltype and attention head. Create a network of genes for each cell type.
# 
# ### Choosen classifier :  Sex classifier
# ### No of Attention Heads :  4 Heads
# ### No of layers : 6 Encoder Layers
# ### Cell Types :  113

# %%
!nvidia-smi

# %%
#Importing the necessary libraries
import scanpy as sc
import os
from geneformer import TranscriptomeTokenizer, DataCollatorForCellClassification
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import datasets
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch
import pickle
from matplotlib.colors import LogNorm, PowerNorm
import networkx as nx
from networkx.algorithms import community as nx_comm
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable




# %% [markdown]
# ## Function/Method to build a record dictionary

# %%
def average_attentions_by_cell_type(
    trainer,
    dataset,
    pad_token_id: int = 0, #If attention matrix between the cells of a cell type is not of the same size, then we will pad the smaller attention matrix with the pad_token_id.
    max_per_type: int = 1000, #max cells per cell type.
    max_total: int = 10000, #max_total cells.
):
    """
    Returns
    -------
    rec: list of dicts (one per cell type) rec item: {'cell_type': str, 'input_ids': List[List[int]], 'attentions': Tuple[Tensor(num_heads, L, L) per layer]}
    rec_keys: list[int] variable to index rec
    structure of record variable:
    rec[cell_type] = {'cell_type': cell_type_value,
            'input_ids': record['input_ids'],
            'attentions': average_attentions}
    """
    model = trainer.model
    model.eval()# setting the model to eval mode to output the attention from the model
    device = model.device #gpu or cpu
    avg_attentions_by_cell_type = {} #dictionary to store the average attention matrix for each cell type
    total_included = 0 #counter to keep track of the total number of cells included

    with torch.no_grad():#no gradient calculation cause not neede to calculate the gradient for the attention matrix
        for sample in dataset:#iterating over the dataset 1 sample at a time
            if total_included >= max_total:
                break

            cell_type_value = sample.get('cell_type', 'unknown')#getting the cell type of the sample
            record = avg_attentions_by_cell_type.get(cell_type_value)#getting the record of the average attention matrix for the cell type
            if record is None:#if the record is not present in the dictionary, then create a new record
                record = {
                    'count': 0,
                    'input_ids': [],  # list of input_ids for this cell type
                    'attn_sum': None,  # tuple of summed attention tensors per layer
                    'seq_len': None,   # ensure consistent sequence length when averaging
                }
                avg_attentions_by_cell_type[cell_type_value] = record

            if record['count'] >= max_per_type:
                continue

            ids = sample['input_ids']#getting the input_ids of the sample
            if isinstance(ids, torch.Tensor):#if the input_ids is a tensor
                ids_list = ids.tolist()#converting the tensor to a list
            else:
                ids_list = ids

            ids_t = torch.tensor([ids_list], dtype=torch.long, device=device)#converting the input_ids to a tensor
            attention_mask = (ids_t != pad_token_id).long()#creating the attention mask
            out = model(input_ids=ids_t, attention_mask=attention_mask, output_attentions=True)#getting the output of the model

            # Tuple over layers; each tensor (num_heads, L, L)
            att_layers = tuple(att.squeeze(0).cpu() for att in out.attentions)#getting the attention layers
            seq_len = att_layers[0].shape[-1]#getting the sequence length

            if record['seq_len'] is None:#if the sequence length is not present in the record
                record['seq_len'] = seq_len#then add the sequence length to the record
            if seq_len != record['seq_len']:
                # Skip samples with mismatched sequence lengths for this cell type
                continue

            if record['attn_sum'] is None:#if the attention sum is not present in the record
                record['attn_sum'] = tuple(att.clone() for att in att_layers)#then add the attention layers to the record
            else:#if the attention sum is present in the record
                record['attn_sum'] = tuple(s + a for s, a in zip(record['attn_sum'], att_layers))#then add the attention layers to the record

            record['count'] += 1
            record['input_ids'].append(ids_list)
            total_included += 1

    # Loop to build the record of the average attention matrix for each cell type.
    rec = []
    for cell_type_value, record in average_attentions_by_cell_type.items():
        if record['count'] == 0 or record['attn_sum'] is None:
            continue
        avg_attentions = tuple(s / record['count'] for s in record['attn_sum'])
        rec.append({
            'cell_type': cell_type_value,
            'input_ids': record['input_ids'],
            'attentions': avg_attentions,
        })

    rec_keys = list(range(len(rec)))
    return rec, rec_keys

# %% [markdown]
# ## Code to plot

# %%

# Compute communities from attention matrix, similar to Mod 1 above
def _compute_communities_from_attention(att_matrix,
                                        token_ids,
                                        threshold: float,
                                        undirected: bool = True):
    """
    Build a graph from attention using >= threshold on the averaged bidirectional
    weight (0.5 * (att[i,j] + att[j,i])) for undirected graphs. For directed graphs,
    threshold raw directional weights. Returns (graph, node_to_comm, node_colors).
    """
    att = np.asarray(att_matrix) # convert the attention matrix to a numpy array
    if att.ndim != 2: # check if the attention matrix is the wrong shape
        raise ValueError("att_matrix must be 2D (L x L)")
    L = att.shape[0]
    if len(token_ids) != L: # check if the token ids length matches the attention size
        raise ValueError("token_ids length must match attention size")

    G = nx.Graph() if undirected else nx.DiGraph() # create a graph
    G.add_nodes_from(token_ids) # add the token ids to the graph

    if undirected:
        sym = 0.5 * (att + att.T) # averaging the attention matrix
        A = sym >= threshold # thresholding the attention matrix
        np.fill_diagonal(A, False) # setting the diagonal to False
        tri = np.argwhere(np.triu(A, 1)) # getting the upper triangular matrix
        for s, t in tri: 
            w = float(sym[s, t]) # getting the weight of the edge
            G.add_edge(token_ids[s], token_ids[t], weight=w) # adding the edge to the graph
    else:
        A = att >= threshold # thresholding the attention matrix
        np.fill_diagonal(A, False) # setting the diagonal to False
        idx = np.argwhere(A)
        for s, t in idx:
            w = float(att[s, t]) # getting the weight of the edge
            G.add_edge(token_ids[s], token_ids[t], weight=w) # adding the edge to the graph

    if G.number_of_edges() > 0 and G.number_of_nodes() > 0: # check if the graph has edges and nodes
        comms = list(nx_comm.greedy_modularity_communities(G, weight="weight")) # getting the communities
    else:
        comms = [set(G.nodes())] if G.number_of_nodes() > 0 else []

    node_to_comm = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node_to_comm[n] = i

    num_clusters = max(1, max(node_to_comm.values(), default=0) + 1)
    cmap_nodes = cm.get_cmap("tab20", num_clusters) # colormap for the nodes
    node_colors = [cmap_nodes(node_to_comm.get(n, 0)) for n in G.nodes()] # getting the colors for the nodes

    return G, node_to_comm, node_colors

# Compute hierarchical clustering order from raw attention, similar to Mod 1 above
def _cluster_order_from_raw_attention(att_matrix,
                                      method: str = "average",
                                      metric: str = "euclidean"):
    """
    Compute hierarchical clustering order (row and col) from the RAW attention matrix.
    Returns two lists of indices: row_order, col_order.
    """
    att = np.asarray(att_matrix) # convert the attention matrix to a numpy array
    try:
        cm_fig = sns.clustermap(
            att,
            method=method,
            metric=metric,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            cbar_pos=None,
        )
        row_order = cm_fig.dendrogram_row.reordered_ind if cm_fig.dendrogram_row is not None else list(range(att.shape[0])) # getting the row order to create the dendrogram
        col_order = cm_fig.dendrogram_col.reordered_ind if cm_fig.dendrogram_col is not None else list(range(att.shape[1])) # getting the column order to create the dendrogram
        plt.close(cm_fig.fig) # closing the figure
    except Exception:
        row_order = list(range(att.shape[0]))
        col_order = list(range(att.shape[1]))
    return row_order, col_order

def visualize_layer(rec,
                    record_idx: int = 0,
                    layer_idx: int = 0,
                    threshold: float = 0.0275,
                    max_heads: int = 6,
                    cmap: str = "Spectral",
                    undirected: bool = True,
                    seed: int = 42,
                    out_prefix: str | None = None):
    """
    For a given layer, produce for each head (up to max_heads) a figure with 3 panels:
      1) Edge-weight plot (sorted of averaged weights)
      2) Clustered network (greedy modularity colors)
      3) RAW attention heatmap reordered by hierarchical clustering; add community color bands
    """
    ids = rec[record_idx]["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, np.ndarray)):
        token_ids = list(ids[0])
    else:
        token_ids = list(ids)

    heads = rec[record_idx]["attentions"][layer_idx]
    arr = heads.detach().cpu().numpy() if hasattr(heads, "detach") else np.array(heads)
    if arr.ndim == 2:
        arr = arr[None, ...]

    n_heads = min(max_heads, arr.shape[0])

    for h in range(n_heads):
        att = arr[h]

        # Network + communities for colors
        G, node_to_comm, node_colors = _compute_communities_from_attention(
            att, token_ids, threshold=threshold, undirected=undirected
        )
        pos = nx.spring_layout(G, seed=seed) if G.number_of_nodes() > 0 else {}
        edge_weights = [d.get("weight", 0.0) for _, _, d in G.edges(data=True)]
        edge_weights_sorted = sorted(edge_weights, reverse=True)

        # 1) Edge-weight plot (sorted)
        # Plot the sorted edge weights
        # (same as Mod 1 above)
        fig, axes = plt.subplots(1, 3, figsize=(24, 10))
        axes[0].set_title(
            f"Edges: {G.number_of_edges()}  Nodes: {G.number_of_nodes()}  Clusters: {len(set(node_to_comm.values())) if G.number_of_nodes()>0 else 0}  (threshold={threshold})"
        )
        if len(edge_weights_sorted) == 0:
            axes[0].text(0.5, 0.5, "No edges", ha="center", va="center")
        else:
            x = np.arange(1, len(edge_weights_sorted) + 1)
            axes[0].plot(x, edge_weights_sorted, lw=1.2)
        axes[0].set_xlabel("Edge index (sorted)")
        axes[0].set_ylabel("Edge weight (avg attention)")
        axes[0].grid(True, alpha=0.3)

        # 2) Clustered network (greedy modularity colors)
        num_clusters = len(set(node_to_comm.values())) if G.number_of_nodes() > 0 else 0
        axes[1].set_title(f"Layer {layer_idx} - Head {h+1}: network (clusters={num_clusters}, threshold={threshold})")
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            axes[1].text(0.5, 0.5, "Empty graph", ha="center", va="center")
            axes[1].axis("off")
        else:
            nx.draw_networkx_nodes(G, pos, node_size=60, node_color=node_colors, alpha=0.85, ax=axes[1])
            nx.draw_networkx_edges(G, pos, width=0.6, alpha=0.5, ax=axes[1])
            axes[1].axis("off")

        # 3) RAW attention heatmap reordered by hierarchical clustering
        # Sort colors in the same order as the heatmap clustering
        row_order, col_order = _cluster_order_from_raw_attention(att)
        att_ordered = att[np.ix_(row_order, col_order)]

        # Map token -> color for original order
        num_clusters = max(1, (max(node_to_comm.values()) + 1) if node_to_comm else 1)
        cmap_nodes = cm.get_cmap("tab20", num_clusters)
        color_by_token = {n: cmap_nodes(node_to_comm.get(n, 0)) for n in token_ids}

        # Sort by color instead of token_ids order
        token_color_pairs = [(token_ids[i], color_by_token[token_ids[i]]) for i in range(len(token_ids))]
        token_color_pairs.sort(key=lambda x: x[1])  # Sort by color

        # Create mapping from original index to color-sorted index
        color_sorted_indices = [token_ids.index(pair[0]) for pair in token_color_pairs]

        # Now, reorder the heatmap and color bands according to color order
        att_color_ordered = att[np.ix_(color_sorted_indices, color_sorted_indices)]
        row_colors_sorted = [color_by_token[token_ids[i]] for i in color_sorted_indices]
        col_colors_sorted = [color_by_token[token_ids[j]] for j in color_sorted_indices]

        # Plot the heatmap ordered by color
        im = axes[2].imshow(att_color_ordered, cmap=cmap, aspect="auto", vmin=0, vmax=float(np.quantile(att, 0.99)))
        # Move the title above the color bands so it's not covered
        axes[2].set_xticks([])
        axes[2].set_yticks([])

        # Add community color bands (single band from network communities)
        # The color bands are now aligned to the color-sorted order
        row_colors = np.array(row_colors_sorted)[None, :, :]
        col_colors = np.array(col_colors_sorted)[:, None, :]

        divider = make_axes_locatable(axes[2])
        ax_col = divider.append_axes("top", size=0.25, pad=0.02)
        ax_row = divider.append_axes("left", size=0.25, pad=0.02)
        ax_col.imshow(row_colors, aspect="auto")
        ax_col.set_axis_off()
        ax_row.imshow(col_colors, aspect="auto")
        ax_row.set_axis_off()

        # Set the title above the color bands using the main axes' figure
        # Use fig.suptitle for global title, or axes[2].set_title with y>1
        axes[2].set_title(f"Attention heatmap (ordered by color, threshold={threshold})", y=1.18)

        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.set_label("attention")

        plt.tight_layout()
        name = (
            f"{out_prefix}_layer{layer_idx}_head{h+1}_summary.png"
            if out_prefix
            else f"layer{layer_idx}_head{h+1}_summary.png"
        )
        plt.savefig(name, dpi=300, bbox_inches="tight")
        plt.show()

def visualize_all_layers(rec,
                         record_idx: int = 0,
                         threshold: float = 0.0275,
                         max_heads: int = 6,
                         cmap: str = "Spectral",
                         undirected: bool = True,
                         seed: int = 42,
                         out_prefix: str | None = None):
    num_layers = len(rec[record_idx]["attentions"]) if rec and rec[record_idx].get("attentions") is not None else 0
    for layer_idx in range(num_layers):
        visualize_layer(
            rec,
            record_idx=record_idx,
            layer_idx=layer_idx,
            threshold=threshold,
            max_heads=max_heads,
            cmap=cmap,
            undirected=undirected,
            seed=seed,
            out_prefix=out_prefix,
        )


# %%
# Old Plot method for visualizing all attention heads in each layer as a grid of heatmaps
def plot_all_layers_as_head_grids(rec, record_idx: int = 0, max_heads: int = 6, cmap: str = 'Spectral'):
    # Get the number of layers from the record
    num_layers = len(rec[record_idx]['attentions'])
    for layer_idx in range(num_layers):
        # Extract the attention heads for this layer
        heads = rec[record_idx]['attentions'][layer_idx]
        # Convert to numpy array (handle torch tensor if needed)
        arr = heads.detach().cpu().numpy() if hasattr(heads, 'detach') else np.array(heads)
        # If only one head, add a new axis to keep consistent shape
        if arr.ndim == 2:
            arr = arr[None, ...]
        # Limit to max_heads (e.g., 6)
        n = min(max_heads, arr.shape[0])
        # Compute vmax for color scaling: 99th percentile of all upper triangle values across heads
        uppers = []
        for i in range(n):
            h = arr[i]
            if h.shape[0] > 1:
                # Take upper triangle (excluding diagonal) for square matrices
                uppers.append(h[np.triu_indices_from(h, 1)])
            else:
                # For 1x1 or 1D, just flatten
                uppers.append(h.ravel())
        all_vals = np.concatenate(uppers) if uppers else np.array([0.0])
        vmax = float(np.quantile(all_vals, 0.99)) if all_vals.size else None

        # Create a 2x2 grid of subplots for up to 4 heads
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.ravel()
        # Hide all axes by default
        for ax in axes:
            ax.axis('off')
        # Plot each head's attention as a heatmap
        for i in range(n):
            im = axes[i].imshow(arr[i], cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
            axes[i].set_title(f'Layer {layer_idx} – Head {i}')
            axes[i].set_xticks([]); axes[i].set_yticks([])
            plt.colorbar(im, ax=axes[i], fraction=0.05, pad=0.04)
        plt.tight_layout()
        
        # Save the figure as a high quality SVG file
        svg_filename = f'attention_layer_{layer_idx}_heads.svg'
        plt.savefig(svg_filename, format='svg', dpi=600, bbox_inches='tight')
        plt.show()


# %% [markdown]
# ## Load the dataset for attention analysis subset of (1000 cells for a celltype)
# ### Dataset balanced for : sex

# %%
dataset_test=datasets.load_from_disk("/hpcfs/users/a1841503/Geneformer/cellnexus/data/sex/cell_type/prepared_tokenized_combined_data/sex_classifier_cellnexus_v2_blood_cell_type_as_covariate_labeled_test.dataset")
dataset_test

# %% [markdown]
# ### Load The Trained Sex Classifier Model

# %%
sex_classifier_model_dir="/hpcfs/users/a1841503/Geneformer/cellnexus/trained_classifiers/sex_classifier_v1_cellnexus/250720190054/250720_geneformer_cellClassifier_sex_classifier_cellnexus/ksplit1/run-e670ee6b/checkpoint-85166"

model_2048 = BertForSequenceClassification.from_pretrained(
    sex_classifier_model_dir,
    num_labels=2 , # or however many classes you have
    ignore_mismatched_sizes=True,
    output_attentions=True
)

# %%
# Model summary 

config = model_2048.config
encoder = model_2048.bert.encoder
embeddings = model_2048.bert.embeddings
classifier = model_2048.classifier
layer0 = encoder.layer[0]
attn = layer0.attention
intermediate = layer0.intermediate
output = layer0.output

def format_linear(module):
    return f"{tuple(module.weight.shape)} (in={module.in_features}, out={module.out_features})"

print("Model:", type(model_2048).__name__)
print(
    "Config:",
    f"d_model={config.hidden_size}, heads={config.num_attention_heads}, "
    f"d_head={config.hidden_size // config.num_attention_heads}, d_ff={config.intermediate_size}, "
    f"seq_max={config.max_position_embeddings}, vocab={config.vocab_size}, layers={len(encoder.layer)}"
)

print(
    "Embeddings:",
    f"word={tuple(embeddings.word_embeddings.weight.shape)}, "
    f"pos={tuple(embeddings.position_embeddings.weight.shape)}, "
    f"types={tuple(embeddings.token_type_embeddings.weight.shape)}, "
    f"ln={embeddings.LayerNorm.normalized_shape}"
)

print(
    "Layer0.Attention:",
    f"Q={format_linear(attn.self.query)}, K={format_linear(attn.self.key)}, "
    f"V={format_linear(attn.self.value)}, out={format_linear(attn.output.dense)}, "
    f"ln={attn.output.LayerNorm.normalized_shape}"
)

print(
    "Layer0.FFN:",
    f"dense={format_linear(intermediate.dense)}, act={intermediate.intermediate_act_fn}"
)

print(
    "Layer0.Output:",
    f"dense={format_linear(output.dense)}, ln={output.LayerNorm.normalized_shape}"
)

print("Classifier:", format_linear(classifier))

total_params = sum(p.numel() for p in model_2048.parameters())
trainable_params = sum(p.numel() for p in model_2048.parameters() if p.requires_grad)
print(f"Params: total={total_params:,}, trainable={trainable_params:,}")


# %%
training_args = TrainingArguments(       # Already points to ./experiments/{timestamp}/checkpoints              # Already points to ./experiments/{timestamp}/logs
    num_train_epochs=1,
    per_device_train_batch_size=4,        
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.1,
    logging_steps=50,
    #evaluation_strategy="steps",
    #save_strategy="epoch",
    #load_best_model_at_end=True,
    learning_rate=1e-5,
    gradient_checkpointing=False,                # Using your timestamp-based run name
    report_to=["tensorboard"],
    # Additional suggested parameters that work with your logging:
    metric_for_best_model="eval_loss",
    save_total_limit=2,                 # Only keeps last 2 checkpoints in your checkpoints directory
    fp16=True,
    # Logging specific parameters
    logging_first_step=True,            # Log the first training step
    logging_strategy="steps",           # Log every n steps
)

# %%
token_dict=pickle.load(open("/hpcfs/users/a1841503/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", "rb"))
trainer_sex = Trainer(
    model=model_2048,
    args=training_args,
    train_dataset=dataset_test,
    data_collator=DataCollatorForCellClassification(token_dictionary=token_dict),
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
)

# %%
# Convert to list first, then use set to get unique values
set(dataset_test['cell_type'])
cell_type_value_counts = pd.Series(dataset_test['cell_type']).value_counts()

# %% [markdown]
# ## Output the attention matrix for the 1k cells 

# %%
# Create the average attention matrix for the 1k cells per cell type and 10k cells samples
rec, rec_keys = average_attentions_by_cell_type(
    trainer_sex,
    dataset_test,
    pad_token_id=0,
    max_per_type=1000,
    max_total=10000,
)

# %%
# Save rec and rec_keys to a pickle file
save_dir = "/hpcfs/users/a1841503/attention_matrix_geneformer_cellnexus/results/"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "rec_and_rec_keys.pkl")
with open(save_path, 'wb') as f:
    pickle.dump({'rec': rec, 'rec_keys': rec_keys}, f)

print(f"rec and rec_keys saved to: {save_path}")


# %%
# Load rec and rec_keys from the pickle file
save_dir = "/hpcfs/users/a1841503/attention_matrix_geneformer_cellnexus/results/"
save_path = os.path.join(save_dir, "rec_and_rec_keys.pkl")

with open(save_path, 'rb') as f:
    loaded_data = pickle.load(f)
    rec = loaded_data['rec']
    rec_keys = loaded_data['rec_keys']

print(f"rec and rec_keys loaded from: {save_path}")


# %%
print(len(rec), rec[0]['cell_type'])
print('num_layers:', len(rec[0]['attentions']))
print('layer0 shape:', rec[0]['attentions'][0].shape)  # (num_heads, L, L)

# %% [markdown]
# ## Plot the average matrix produced

# %%
# Set up spectral color palette similar to ggplot2's scale_color_brewer(type="seq", palette="Spectral")
# Create spectral palette similar to ggplot2's ColorBrewer Spectral
spectral_colors = ['#9e0142', '#d53e4f', '#f46d43', '#fdae61', '#fee08b', 
                   '#e6f598', '#abdda4', '#66c2a5', '#3288bd', '#5e4fa2']
spectral_cmap = ListedColormap(spectral_colors)

# Set this as default colormap for plotting
plt.rcParams['image.cmap'] = 'Spectral'

# %%
#Calling the method to plot the attention heads for each layer
plot_all_layers_as_head_grids(rec, record_idx=0, max_heads=6, cmap='Spectral')


# %%
# Details of the attention matrix
attention_tensor = rec[0]['attentions'][0]
attention_flat = attention_tensor.flatten()

print("Attention Matrix Statistics:")
print(f"Shape: {attention_tensor.shape}")
print(f"Count: {attention_flat.numel()}")
print(f"Mean: {attention_flat.mean().item():.6f}")
print(f"Std: {attention_flat.std().item():.6f}")
print(f"Min: {attention_flat.min().item():.6f}")
print(f"25%: {torch.quantile(attention_flat, 0.25).item():.6f}")
print(f"50% (Median): {torch.median(attention_flat).item():.6f}")
print(f"75%: {torch.quantile(attention_flat, 0.75).item():.6f}")
print(f"Max: {attention_flat.max().item():.6f}")

# %%
#Minimal threshold
min_threshold=attention_flat.min().item()
visualize_all_layers(rec, record_idx=0, threshold=min_threshold)

# %%
medium_threshold=attention_flat.median().item()
medium_threshold

# %%
medium_threshold=attention_flat.median().item()
#Medium threshold
visualize_all_layers(rec, record_idx=0, threshold=0.00028)

# %%
compute_global_color_maps(rec, record_idx=0, layer_idx=0, head_idx=0,
                          threshold_A=0.0, threshold_B=0.275)
VARIABLE_A, VARIABLE_B, SORTED_VARIABLE_A, SORTED_VARIABLE_B

# %%
visualize_all_layers(rec, record_idx=0, threshold=0.00028)

# %%
visualize_all_layers(rec, record_idx=0, threshold=0.00028)

# %%
max_threshold=attention_flat.max().item()
#Maximal threshold
visualize_all_layers(rec, record_idx=0, threshold=max_threshold)

# %%
visualize_all_layers(rec, record_idx=0, threshold=0.05)

# %% [markdown]
# ## Build a gene network, thresholding on min attention value.

# %% [markdown]
# ## Graph for 1 celltype and 1 attention head

# %% [markdown]
# 

# %%
type(rec[0]['attentions'])

# %%
len(rec[0]['input_ids'])

# %%
# Create an empty graph
G = nx.Graph()

# Add nodes to the graph using token IDs
for token_id in rec[0]['input_ids']:
    G.add_node(token_id)

# Get attention scores from the first attention head of the first layer
attention_matrix = rec[0]['attentions'][0][0]  # Shape: [seq_len, seq_len] - added [0] to get first head

# Add edges to the graph based on attention scores
# Use a threshold to only add edges for significant attention weights
threshold = 0.0275  # Adjust this threshold as needed
for i in range(len(rec[0]['input_ids'])):
    for j in range(len(rec[0]['input_ids'])):  # Additional for loop as requested
        if i != j and attention_matrix[i, j].item() > threshold:
            # Add edge with attention score as weight
            G.add_edge(rec[0]['input_ids'][i], rec[0]['input_ids'][j], 
                      weight=float(attention_matrix[i, j].item()))

print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")


# %%
# Create an empty graph
G = nx.Graph()

# Add nodes to the graph using token IDs
for token_id in rec[0]['input_ids']:
    G.add_node(token_id)

# Get attention scores from the first attention head of the first layer
attention_matrix = rec[0]['attentions'][0][0]  # Shape: [seq_len, seq_len]

# Add edges to the graph based on attention scores
# Use a threshold to only add edges for significant attention weights
threshold = 0.02  # Adjust this threshold as needed
for i in range(len(rec[0]['input_ids'])):
    for j in range(len(rec[0]['input_ids'])):
        if i != j and attention_matrix[i, j] > threshold:
            # Add edge with attention score as weight
            G.add_edge(rec[0]['input_ids'][i], rec[0]['input_ids'][j], 
                      weight=float(attention_matrix[i, j]))


# %%

# Choose which layer to visualize
layer_idx = 0            # change to 0..(num_layers-1)
threshold = 0.0275       # adjust as needed

# Get token ids
input_ids = rec[0]['input_ids']
token_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
seq_len = len(token_ids)

# Get attention for the chosen layer
layer_attn = rec[0]['attentions'][layer_idx]

# Normalize to numpy array per head: per_head_attn is a list of 2D arrays [seq_len, seq_len]
per_head_attn = []
try:
    arr = layer_attn.detach().cpu().numpy()
    if arr.ndim == 3:  # (heads, seq, seq)
        per_head_attn = [arr[h] for h in range(arr.shape[0])]
    elif arr.ndim == 2:
        per_head_attn = [arr]
    else:
        raise ValueError(f"Unexpected attention shape: {arr.shape}")
except Exception:
    # Handle list/tuple or numpy tensors
    if isinstance(layer_attn, (list, tuple)):
        for head_mat in layer_attn:
            try:
                per_head_attn.append(head_mat.detach().cpu().numpy())
            except Exception:
                try:
                    per_head_attn.append(head_mat.numpy())
                except Exception:
                    per_head_attn.append(np.array(head_mat))
    else:
        arr = np.array(layer_attn)
        if arr.ndim == 3:
            per_head_attn = [arr[h] for h in range(arr.shape[0])]
        elif arr.ndim == 2:
            per_head_attn = [arr]
        else:
            raise ValueError(f"Unexpected attention shape: {arr.shape}")

# Build a graph per head
graphs_by_head = []
for head_idx, att in enumerate(per_head_attn):
    G = nx.Graph()
    G.add_nodes_from(token_ids)
    for i in range(seq_len):
        for j in range(seq_len):
            if i == j:
                continue
            score = att[i, j].item() if hasattr(att[i, j], 'item') else float(att[i, j])
            if score > threshold:
                G.add_edge(token_ids[i], token_ids[j], weight=float(score))
    graphs_by_head.append(G)

print(f"Built {len(graphs_by_head)} graphs for layer {layer_idx} (one per head)")

# Use a common layout across heads for comparability (union of edges)
combined_G = nx.Graph()
combined_G.add_nodes_from(token_ids)
for G in graphs_by_head:
    combined_G.add_edges_from(G.edges())
pos = nx.spring_layout(combined_G, k=0.5, iterations=500, seed=42)

# Draw and save each head's graph
for h, G in enumerate(graphs_by_head, start=1):
    plt.figure(figsize=(18, 12))
    nx.draw_networkx_nodes(G, pos, node_size=60, node_color='blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.6)
    plt.axis("off")
    out_path = f"attention_graph_layer{layer_idx}_head{h}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")

# %%
ids = list(rec[0]['input_ids'])

# %%
len(ids[0])

# %%
import pickle
with open(f"/hpcfs/users/a1841503/Geneformer/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl", "rb") as f:
    token_dictionary_gc30M = pickle.load(f)

# %% [markdown]
# ## Trying to use bertviz to do a visualisation of the attentions and the tokens

# %%
import torch, numpy as np
from bertviz import head_view

rec_idx = 0
id2tok = None  # or a list mapping id->token

# Fix the indexing issue - rec is a list, so we need to access the record first
record = rec[rec_idx]
ids = record['input_ids']

# Handle the case where ids might be nested or have extra dimensions
if isinstance(ids, torch.Tensor):
    ids = ids.squeeze().tolist()
elif isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, torch.Tensor)):
    ids = ids[0] if isinstance(ids[0], list) else ids[0].tolist()

tokens = [str(id2tok[i]) if isinstance(id2tok, list) and 0 <= i < len(id2tok) else str(i) for i in ids] if id2tok is not None else [str(i) for i in ids]

attn = []
for t in record['attentions']:
    a = t if isinstance(t, torch.Tensor) else torch.as_tensor(t)
    a = a.detach().cpu().float()
    if a.dim() == 3:      # (H, L, L) -> (1, H, L, L)
        a = a.unsqueeze(0)
    elif a.dim() != 4:
        raise ValueError(f'Unexpected attention shape: {tuple(a.shape)}')
    attn.append(a)

head_view(attention=attn, tokens=tokens)

# %%
# Fix: convert PyTorch tensors to NumPy before thresholding/diagonal ops

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community as nx_comm

def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.array(x)

def _get_token_ids(rec, record_idx=0):
    ids = rec[record_idx]['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, np.ndarray)):
        return list(ids[0])
    return list(ids)

def _ensure_heads(arr):
    A = _to_numpy(arr)
    return A if A.ndim == 3 else A[None, ...]

def build_graph_from_attention(att, token_ids, threshold=0.0275, undirected=True):
    att = _to_numpy(att)  # ensure numpy
    L = len(token_ids)
    A = att > threshold
    np.fill_diagonal(A, False)

    if undirected:
        M = np.logical_or(A, A.T)
        G = nx.Graph()
        G.add_nodes_from(token_ids)
        tri = np.argwhere(np.triu(M, 1))
        for s, t in tri:
            w = 0.5 * (float(att[s, t]) + float(att[t, s]))
            G.add_edge(token_ids[s], token_ids[t], weight=w)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(token_ids)
        idx = np.argwhere(A)
        for s, t in idx:
            w = float(att[s, t])
            G.add_edge(token_ids[s], token_ids[t], weight=w)
    return G

def cluster_graph(G, method='greedy_modularity', seed=42, weight='weight'):
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return {n: 0 for n in G.nodes()}, [set(G.nodes())]
    if method in ('greedy', 'greedy_modularity'):
        comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))
    elif method in ('lpa', 'label_propagation'):
        comms = list(nx_comm.asyn_lpa_communities(G, weight=weight, seed=seed))
    elif method == 'louvain':
        try:
            import community as community_louvain
            part = community_louvain.best_partition(G.to_undirected(), weight=weight, random_state=seed)
            comm_index = {}
            for n, c in part.items():
                comm_index.setdefault(c, set()).add(n)
            sorted_keys = sorted(comm_index.keys())
            node_to_comm = {n: sorted_keys.index(c) for n, c in part.items()}
            comms = [comm_index[k] for k in sorted_keys]
            return node_to_comm, comms
        except Exception:
            comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))
    else:
        comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))
    node_to_comm = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node_to_comm[n] = i
    return node_to_comm, comms

def cluster_layer_heads(rec, record_idx=0, layer_idx=0, threshold=0.0275,
                        method='greedy_modularity', max_heads=6, seed=42,
                        undirected=True, out_prefix=None, save_csv=True, plot=True):
    token_ids = _get_token_ids(rec, record_idx)
    arr = _ensure_heads(rec[record_idx]['attentions'][layer_idx])  # numpy (H,L,L)
    n_heads = min(max_heads, arr.shape[0])

    # shared layout for consistent comparison
    combined = nx.Graph()
    combined.add_nodes_from(token_ids)
    for h in range(n_heads):
        att = arr[h]
        A = att > threshold
        np.fill_diagonal(A, False)
        M = np.logical_or(A, A.T) if undirected else A
        tri = np.argwhere(np.triu(M, 1)) if undirected else np.argwhere(M)
        for s, t in tri:
            combined.add_edge(token_ids[s], token_ids[t])
    pos = nx.spring_layout(combined, k=0.5, iterations=500, seed=seed)

    results = {'layer': layer_idx, 'method': method, 'threshold': threshold, 'heads': []}
    csv_rows = []

    for h in range(n_heads):
        G = build_graph_from_attention(arr[h], token_ids, threshold=threshold, undirected=undirected)
        tokens_present = sorted([n for n, d in G.degree() if d > 0], key=lambda x: (isinstance(x, str), x))
        node_to_comm, comms = cluster_graph(G, method=method, seed=seed, weight='weight')

        if plot:
            if len(G) == 0:
                fig, ax = plt.subplots(figsize=(10, 7))
                ax.axis('off')
                ax.set_title(f'Layer {layer_idx} – Head {h+1}: empty (thr={threshold})')
                plt.tight_layout()
                name = f"{out_prefix}_layer{layer_idx}_head{h+1}_communities.png" if out_prefix else f"layer{layer_idx}_head{h+1}_communities.png"
                plt.savefig(name, dpi=300, bbox_inches='tight')
                plt.show()
            else:
                import matplotlib.cm as cm
                cmap = cm.get_cmap('tab20', max(1, len(comms)))
                node_colors = [cmap(node_to_comm[n]) for n in G.nodes()]
                plt.figure(figsize=(12, 9))
                nx.draw_networkx_nodes(G, pos, node_size=60, node_color=node_colors, alpha=0.85)
                nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.4)
                plt.axis("off")
                plt.title(f'Layer {layer_idx} – Head {h+1}: Communities ({method}, thr={threshold})')
                plt.tight_layout()
                name = f"{out_prefix}_layer{layer_idx}_head{h+1}_communities.png" if out_prefix else f"layer{layer_idx}_head{h+1}_communities.png"
                plt.savefig(name, dpi=300, bbox_inches='tight')
                plt.show()

        for node in G.nodes():
            deg = G.degree(node)
            comm = node_to_comm.get(node, -1)
            csv_rows.append([layer_idx, h + 1, node, comm, deg])

        results['heads'].append({
            'head': h + 1,
            'tokens_present': tokens_present,
            'communities': [sorted(list(c)) for c in comms],
            'node_to_community': node_to_comm
        })

    if save_csv and csv_rows:
        import csv
        csv_name = f"{out_prefix or 'communities'}_layer{layer_idx}.csv"
        with open(csv_name, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['layer', 'head', 'token_id', 'community_id', 'degree'])
            w.writerows(csv_rows)

    return results

def tokens_in_each_head_all_layers(rec, record_idx=0, threshold=0.0275, undirected=True):
    out = {}
    num_layers = len(rec[record_idx]['attentions'])
    token_ids = _get_token_ids(rec, record_idx)
    for layer_idx in range(num_layers):
        arr = _ensure_heads(rec[record_idx]['attentions'][layer_idx])  # numpy (H,L,L)
        layer_map = {}
        for h in range(arr.shape[0]):
            G = build_graph_from_attention(arr[h], token_ids, threshold=threshold, undirected=undirected)
            present = sorted([n for n, d in G.degree() if d > 0], key=lambda x: (isinstance(x, str), x))
            layer_map[h + 1] = present
        out[layer_idx] = layer_map
    return out



# %%
# Example:
res = cluster_layer_heads(rec, record_idx=0, threshold=0.03,
                           method='greedy_modularity', max_heads=6, seed=42,
                           undirected=True, out_prefix='L0', save_csv=True, plot=True)
tokens_map = tokens_in_each_head_all_layers(rec, record_idx=0, threshold=0.03, undirected=True)
print(tokens_map[0][1])

# %%
# Tables of genes/tokens with attention scores per layer/head, plus cluster membership lists
# Produces:
# - For each layer/head: a pandas DataFrame with per-token attention scores and community id
# - For each layer/head: a list of clusters (each a list of token ids or gene names)
# Optional: write CSVs per head for table and clusters

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_comm


def _to_numpy(x):
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.array(x)


def _get_token_ids(rec, record_idx=0):
    ids = rec[record_idx]["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, np.ndarray)):
        return list(ids[0])
    return list(ids)


def _ensure_heads(att_per_layer):
    A = _to_numpy(att_per_layer)
    return A if A.ndim == 3 else A[None, ...]  # (H, L, L)


def build_graph_from_attention(att, token_ids, threshold=0.03, undirected=True):
    att = _to_numpy(att)  # (L, L)
    L = len(token_ids)
    A = att > threshold
    np.fill_diagonal(A, False)

    if undirected:
        M = np.logical_or(A, A.T)
        W = 0.5 * (att + att.T)  # symmetric weights
        G = nx.Graph()
        G.add_nodes_from(token_ids)
        tri = np.argwhere(np.triu(M, 1))
        for s, t in tri:
            G.add_edge(token_ids[s], token_ids[t], weight=float(W[s, t]))
    else:
        G = nx.DiGraph()
        G.add_nodes_from(token_ids)
        idx = np.argwhere(A)
        for s, t in idx:
            G.add_edge(token_ids[s], token_ids[t], weight=float(att[s, t]))
    return G


def cluster_graph(G, method="greedy_modularity", seed=42, weight="weight"):
    if G.number_of_edges() == 0 or G.number_of_nodes() == 0:
        return {n: 0 for n in G.nodes()}, [set(G.nodes())]

    if method in ("greedy", "greedy_modularity"):
        comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))
    elif method in ("lpa", "label_propagation"):
        comms = list(nx_comm.asyn_lpa_communities(G, weight=weight, seed=seed))
    elif method == "louvain":
        try:
            import community as community_louvain  # python-louvain
            part = community_louvain.best_partition(G.to_undirected(), weight=weight, random_state=seed)
            comm_index = {}
            for n, c in part.items():
                comm_index.setdefault(c, set()).add(n)
            ordered = sorted(comm_index.keys())
            node_to_comm = {n: ordered.index(c) for n, c in part.items()}
            comms = [comm_index[k] for k in ordered]
            return node_to_comm, comms
        except Exception:
            comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))
    else:
        comms = list(nx_comm.greedy_modularity_communities(G, weight=weight))

    node_to_comm = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node_to_comm[n] = i
    return node_to_comm, comms


def head_table_and_clusters(att, token_ids, threshold=0.03, method="greedy_modularity",
                            undirected=True, token_to_gene=None):
    """
    att: (L, L) attention for a single head (torch or numpy)
    token_ids: list of length L
    Returns:
      df: DataFrame with per-token scores and community_id
      clusters: list[list] of tokens (or gene names if token_to_gene provided)
    """
    att = _to_numpy(att)
    L = len(token_ids)

    # Raw scores (no threshold)
    out_score_raw = att.sum(axis=1)             # row sum
    in_score_raw = att.sum(axis=0)              # col sum

    # Thresholded masks
    M_dir = att > threshold
    np.fill_diagonal(M_dir, False)

    # Thresholded sums
    out_score_thr = (att * M_dir).sum(axis=1)
    in_score_thr = (att * M_dir).sum(axis=0)

    # Undirected mask/weights for weighted-degree
    M_undir = np.logical_or(M_dir, M_dir.T)
    W_undir = 0.5 * (att + att.T) * M_undir
    deg_edges = M_undir.sum(axis=1).astype(int)  # number of edges above threshold
    score_undirected = W_undir.sum(axis=1)

    # Build graph and cluster
    G = build_graph_from_attention(att, token_ids, threshold=threshold, undirected=undirected)
    node_to_comm, comms = cluster_graph(G, method=method, seed=42, weight="weight")

    # Table
    rows = []
    for i, tok in enumerate(token_ids):
        gene = token_to_gene.get(tok, tok) if token_to_gene else tok
        rows.append({
            "token_id": int(tok) if isinstance(tok, (int, np.integer)) else tok,
            "gene": gene,
            "deg_edges_thr": int(deg_edges[i]),
            "score_out_raw": float(out_score_raw[i]),
            "score_in_raw": float(in_score_raw[i]),
            "score_out_thr": float(out_score_thr[i]),
            "score_in_thr": float(in_score_thr[i]),
            "score_undirected_thr": float(score_undirected[i]),
            "community_id": int(node_to_comm.get(tok, -1))
        })
    df = pd.DataFrame(rows).sort_values(
        ["community_id", "deg_edges_thr", "score_undirected_thr"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    # Cluster lists
    clusters = []
    for cset in comms:
        if token_to_gene:
            clusters.append([token_to_gene.get(t, t) for t in sorted(cset, key=lambda x: (isinstance(x, str), x))])
        else:
            clusters.append(sorted(cset, key=lambda x: (isinstance(x, str), x)))

    return df, clusters


def analyze_attention_tables_and_clusters(rec, record_idx=0, threshold=0.03, method="greedy_modularity",
                                          undirected=True, token_to_gene=None, save_csv=True,
                                          out_prefix="attn"):
    """
    For each layer and head:
      - builds a per-token table of attention scores and community id
      - extracts cluster lists (tokens per community)
      - optionally writes CSVs: {out_prefix}_layer{L}_head{H}_genes.csv and ..._clusters.csv
    Returns nested dict: results[layer_idx][head_idx] = {"table": df, "clusters": list_of_lists}
    """
    token_ids = _get_token_ids(rec, record_idx)
    num_layers = len(rec[record_idx]["attentions"])
    results = {}

    for layer_idx in range(num_layers):
        arr = _ensure_heads(rec[record_idx]["attentions"][layer_idx])  # (H, L, L)
        results[layer_idx] = {}
        for h in range(arr.shape[0]):
            df, clusters = head_table_and_clusters(
                arr[h], token_ids, threshold=threshold, method=method,
                undirected=undirected, token_to_gene=token_to_gene
            )
            results[layer_idx][h + 1] = {"table": df, "clusters": clusters}

            if save_csv:
                df_name = f"{out_prefix}_layer{layer_idx}_head{h+1}_genes.csv"
                cl_name = f"{out_prefix}_layer{layer_idx}_head{h+1}_clusters.csv"
                df.to_csv(df_name, index=False)
                # clusters CSV: one row per cluster with size and tokens
                with open(cl_name, "w") as f:
                    f.write("cluster_id,cluster_size,tokens\n")
                    for cid, toks in enumerate(clusters):
                        toks_joined = ";".join(map(str, toks))
                        f.write(f"{cid},{len(toks)},{toks_joined}\n")

    return results

# %%
# Example usage:
# token_to_gene = {123: "TP53", 456: "EGFR", ...}  # optional
results = analyze_attention_tables_and_clusters(
     rec, record_idx=0, threshold=0.03, method="greedy_modularity",
     undirected=True, save_csv=True, out_prefix="gene_heads"
)
# Example access:
df_L0_H1 = results[0][1]["table"]
clusters_L0_H1 = results[0][1]["clusters"]
clusters_L0_H2 = results[0][2]["clusters"]
df_L0_H2 = results[0][2]["table"]
clusters_L0_H3 = results[0][3]["clusters"]
df_L0_H3 = results[0][3]["table"]
clusters_L0_H4 = results[0][4]["clusters"]
df_L0_H4 = results[0][4]["table"]


# %%
# Minimal tables and clusters per layer/head
# Assumes: rec[record_idx]['attentions'][layer] is (H,L,L) tensor/ndarray; rec[record_idx]['input_ids'] is (L,)
import pickle, numpy as np, pandas as pd, networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

def load_token_to_gene(pkl_path):
    return pickle.load(open(pkl_path, 'rb'))  # assumes {token_id:int -> gene:str}

def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.array(x)

def attention_tables_and_clusters(rec, record_idx=0, threshold=0.03, token_to_gene=None, save_csv=False, out_prefix="attn"):
    token_ids = rec[record_idx]['input_ids']
    token_ids = token_ids.tolist() if hasattr(token_ids, 'tolist') else list(token_ids)
    num_layers = len(rec[record_idx]['attentions'])
    results = {}

    for L in range(num_layers):
        A = _np(rec[record_idx]['attentions'][L])  # (H,L,L) or (L,L)
        if A.ndim == 2: A = A[None, ...]
        H = A.shape[0]
        results[L] = {}
        for h in range(H):
            att = A[h]  # (L,L)
            M = att > threshold
            np.fill_diagonal(M, False)
            MU = np.logical_or(M, M.T)
            WU = 0.5 * (att + att.T) * MU

            G = nx.Graph()
            G.add_nodes_from(token_ids)
            idx = np.argwhere(np.triu(MU, 1))
            for s, t in idx:
                G.add_edge(token_ids[s], token_ids[t], weight=float(WU[s, t]))

            comms = list(greedy_modularity_communities(G, weight='weight')) if G.number_of_edges() else [set(G.nodes())]
            node_to_comm = {n: i for i, c in enumerate(comms) for n in c}

            rows = []
            for i, tok in enumerate(token_ids):
                gene = token_to_gene.get(tok, tok) if token_to_gene else tok
                rows.append({
                    "token_id": int(tok) if isinstance(tok, (int, np.integer)) else tok,
                    "gene": gene,
                    "edges_thr": int(MU[i].sum()),
                    "score_undirected_thr": float(WU[i].sum()),
                    "community_id": int(node_to_comm.get(tok, -1)),
                })
            df = pd.DataFrame(rows).sort_values(["community_id", "edges_thr", "score_undirected_thr"], ascending=[True, False, False])

            clusters = []
            for c in comms:
                items = [token_to_gene.get(t, t) for t in c] if token_to_gene else list(c)
                clusters.append(sorted(items, key=lambda x: (isinstance(x, str), x)))

            results[L][h+1] = {"table": df, "clusters": clusters}

            if save_csv:
                df.to_csv(f"{out_prefix}_layer{L}_head{h+1}_genes.csv", index=False)
                with open(f"{out_prefix}_layer{L}_head{h+1}_clusters.csv", "w") as f:
                    f.write("cluster_id,cluster_size,tokens\n")
                    for cid, toks in enumerate(clusters):
                        f.write(f"{cid},{len(toks)},\"{';'.join(map(str, toks))}\"\n")
    return results



# %%
df_L0_H1

# %%
df_L0_H1 = results[0][1]["table"]
clusters_L0_H1 = results[0][1]["clusters"]
print("Head 1 - Table:")
display(df_L0_H1)
print("\nHead 1 - Clusters:")
print(clusters_L0_H1)

df_L0_H2 = results[0][2]["table"]
clusters_L0_H2 = results[0][2]["clusters"]
print("\nHead 2 - Table:")
display(df_L0_H2)
print("\nHead 2 - Clusters:")
print(clusters_L0_H2)

df_L0_H3 = results[0][3]["table"]
clusters_L0_H3 = results[0][3]["clusters"]
print("\nHead 3 - Table:")
display(df_L0_H3)
print("\nHead 3 - Clusters:")
print(clusters_L0_H3)

df_L0_H4 = results[0][4]["table"]
clusters_L0_H4 = results[0][4]["clusters"]
print("\nHead 4 - Table:")
display(df_L0_H4)
print("\nHead 4 - Clusters:")
print(clusters_L0_H4)



# %%

# If you already loaded it as `token_to_gene` but it's actually gene->id:
token_to_gene = pickle.load(open('/hpcfs/users/a1841503/Geneformer/geneformer/token_dictionary_gc104M.pkl','rb'))
ensembl_to_gene = pickle.load(open('/hpcfs/users/a1841503/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl','rb')) 
gene_to_id = token_to_gene  # rename for clarity
id_to_gene = {int(v): str(k) for k, v in gene_to_id.items()}
ensembel_to_gene = ensembl_to_gene
gene_to_ensembl = {str(v): str(k) for k, v in ensembl_to_gene.items()}
# Minimal mappers
def tokens_to_genes(tokens, id2gene):
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if isinstance(tokens, (list, tuple)) and tokens and isinstance(tokens[0], (list, tuple)):
        tokens = tokens[0]
    out = []
    seq = tokens if isinstance(tokens, (list, tuple)) else [tokens]
    for t in seq:
        if hasattr(t, 'item'):
            t = t.item()
        try:
            t = int(t)
        except Exception:
            pass
        out.append(id2gene.get(t, id2gene.get(str(t), t)))
    return out

def clusters_to_genes(clusters, id2gene):
    return [tokens_to_genes(c, id2gene) for c in clusters]

# Quick checks / usage
# print(id_to_gene.get(4))  # -> 'ENSG00000000003'
# genes_seq0 = tokens_to_genes(rec[0]['input_ids'], id_to_gene)
# genes_in_head4 = clusters_to_genes(clusters_L0_H4, id_to_gene)

# %%
# Build both maps, then map tokens -> Ensembl -> gene name (simple)
token_to_ensembl = pickle.load(open('/hpcfs/users/a1841503/Geneformer/geneformer/token_dictionary_gc104M.pkl','rb'))
ensembl_to_gene = pickle.load(open('/hpcfs/users/a1841503/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl','rb')) 
to_ens = lambda x: isinstance(x, str) and x.startswith('ENS')

# token_to_ensembl may be {ENSG -> token_id} or {token_id -> ENSG}
id_to_ensembl = ({int(v): str(k) for k, v in token_to_ensembl.items()} 
                 if any(to_ens(k) for k in token_to_ensembl) 
                 else {int(k): str(v) for k, v in token_to_ensembl.items()})

# ensembl_to_gene may be {ENSG -> GENE} or {GENE -> ENSG}
ens_to_gene = ( {str(k): str(v) for k, v in ensembl_to_gene.items()} 
                if any(to_ens(k) for k in ensembl_to_gene) 
                else {str(v): str(k) for k, v in ensembl_to_gene.items()} )

# final: token_id -> gene_name (fallback to ENSG if name missing)
token_to_gene = {tid: ens_to_gene.get(ens, ens) for tid, ens in id_to_ensembl.items()}

def tokens_to_names(tokens):
    if hasattr(tokens, 'tolist'):
        tokens = tokens.tolist()
    if isinstance(tokens, (list, tuple)) and tokens and isinstance(tokens[0], (list, tuple)):
        tokens = tokens[0]
    if not isinstance(tokens, (list, tuple)):
        tokens = [tokens]
    out = []
    for t in tokens:
        if hasattr(t, 'item'):
            t = t.item()
        try:
            tid = int(t)
        except Exception:
            tid = t
        gene = token_to_gene.get(tid)
        if gene is None:
            ens = id_to_ensembl.get(tid)
            gene = ens_to_gene.get(ens, ens) if ens is not None else tid
        out.append(gene)
    return out

# Examples:
genes_seq0 = tokens_to_names(rec[0]['input_ids'])
genes_in_head4 = [tokens_to_names(cluster) for cluster in clusters_L0_H4]

# %%
# Use your existing id_to_ensembl, and also map Ensembl -> gene name

def ens_clean(e):
    return str(e).split('.', 1)[0]  # drop Ensembl version if present

# Ensure ensembl_to_gene is Ensembl -> gene_name (invert if it's gene_name -> Ensembl)
ens_to_gene = (ensembl_to_gene if any(str(k).startswith('ENS') for k in ensembl_to_gene.keys())
               else {str(v): str(k) for k, v in ensembl_to_gene.items()})
# Normalize keys (strip version)
ens_to_gene = {ens_clean(k): v for k, v in ens_to_gene.items()}

def to_list(x):
    x = x.tolist() if hasattr(x, 'tolist') else x
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], (list, tuple)):
        x = x[0]
    return list(x) if not isinstance(x, list) else x

def tokens_to_gene_names(tokens):
    toks = to_list(tokens)
    names = []
    for t in toks:
        if hasattr(t, 'item'):
            t = t.item()
        try:
            tid = int(t)
        except Exception:
            tid = t
        ens = id_to_ensembl.get(tid)  # you already built this
        gene = ens_to_gene.get(ens_clean(ens), ens) if ens is not None else tid
        names.append(gene)
    return names

# Example:
gene_names_seq0 = tokens_to_gene_names(rec[0]['input_ids'])
gene_names_from_cluster = tokens_to_gene_names(clusters_L0_H4)

# %%
ensembl_to_gene = pickle.load(open('/hpcfs/users/a1841503/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl','rb')) 

# %%
list(ensembl_to_gene.items())[0]

# %%
gene_names_from_cluster

# %%
genes_seq0 = tokens_to_genes(rec[0]['input_ids'], id_to_gene)
genes_in_head4 = clusters_to_genes(clusters_L0_H4, id_to_gene)

# %%
genes_in_head4

# %%
import pickle

# Load your dict (gene_name -> ensembl_id)
with open('/hpcfs/users/a1841503/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl', 'rb') as f:
    gene_to_ensembl = pickle.load(f)

# Invert to ensembl_id -> gene_name (keeps the last gene if duplicates exist)
ensembl_to_gene = {ensembl_id: gene for gene, ensembl_id in gene_to_ensembl.items()}

def ensembl_ids_to_gene_names(ensembl_ids, keep_unmapped=True):
    # keep_unmapped=True returns the original ID when not found; False returns None
    if keep_unmapped:
        return [ensembl_to_gene.get(eid, eid) for eid in ensembl_ids]
    return [ensembl_to_gene.get(eid) for eid in ensembl_ids]

genes = ensembl_ids_to_gene_names(gene_names_from_cluster)
print(genes)

# %%
genes

# %%
# %matplotlib inline
# %config InlineBackend.figure_format='retina' # mac
%load_ext autoreload
%autoreload 2
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt

# %%
gp.__version__

# %%
from gseapy import Biomart
bm = Biomart()

# %%
names = gp.get_library_name()

# %%
yeast = gp.get_library_name(organism='Yeast')

# %%
enr = gp.enrichr(gene_list=genes, 
                 gene_sets=['KEGG_2016','KEGG_2013'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )

# %%
enr.results.head(100)

# %%
enr.results.describe()

# %%
enr2 = gp.enrichr(gene_list=genes, # or "./tests/data/gene_list.txt",
                 gene_sets=['MSigDB_Hallmark_2020','KEGG_2021_Human'],
                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                 outdir=None, # don't write to disk
                )


# %%
# Number of rows in the dataframe
enr2.results.shape[0]


# %%
enr2.results.describe()

# %%
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import cm
from networkx.algorithms import community as nx_comm
from typing import Callable, Dict, Any, Tuple


def _rgba_to_hex(rgba: Tuple[float, float, float, float]) -> str:
    r, g, b = rgba[:3]
    return f"#{int(round(255*r)):02x}{int(round(255*g)):02x}{int(round(255*b)):02x}"


def _hex_to_rgb01(hex_str: str) -> Tuple[float, float, float]:
    s = hex_str.lstrip("#")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return (r, g, b)


def _resolve_gene_name(
    token_id: Any,
    position: int,
    pos_to_name: list | None,
    id_to_name: Dict[Any, str] | Callable[[Any], str] | None,
) -> str:
    if pos_to_name is not None and 0 <= position < len(pos_to_name):
        name = pos_to_name[position]
        if name is not None:
            return str(name)
    if callable(id_to_name):
        try:
            return str(id_to_name(token_id))
        except Exception:
            return str(token_id)
    if isinstance(id_to_name, dict):
        return str(id_to_name.get(token_id, token_id))
    return str(token_id)


def build_gene_community_table(
    rec,
    record_idx: int = 0,
    layer_idx: int = 0,
    threshold: float = 0.0275,
    max_heads: int = 6,
    undirected: bool = True,
    cmap_name: str = "tab20",
    id_to_name: Dict[Any, str] | Callable[[Any], str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Build a per-head table mapping each gene/token to its community and color.

    Returns
    -------
    (df, meta)
      - df: columns = [record_idx, layer, head, position, token_id, gene_name, community, color_hex]
      - meta: dict with keys {token_ids, gene_names, num_heads, num_tokens}

    Notes
    -----
    - Attempts to infer gene names from the record at the same positions via keys
      like 'gene_names' or 'input_tokens'. If unavailable, uses `id_to_name` mapping
      or falls back to the token id as string.
    - Colors are assigned per head, one color per community using `cmap_name`.
    """
    # Token ids from first sequence
    ids = rec[record_idx]["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], (list, np.ndarray)):
        token_ids = list(ids[0])
    else:
        token_ids = list(ids)

    # Try to fetch positional gene names if present in the record
    pos_to_name = None
    if isinstance(rec[record_idx], dict):
        for key in ("gene_names", "genes", "tokens", "token_strs", "input_tokens"):
            maybe = rec[record_idx].get(key)
            if isinstance(maybe, (list, tuple)) and len(maybe) == len(token_ids):
                pos_to_name = list(maybe)
                break

    # Attentions for layer -> numpy (H, L, L)
    heads = rec[record_idx]["attentions"][layer_idx]
    arr = heads.detach().cpu().numpy() if hasattr(heads, "detach") else np.array(heads)
    if arr.ndim == 2:
        arr = arr[None, ...]

    n_heads = min(max_heads, arr.shape[0])
    token_id_to_pos = {tid: i for i, tid in enumerate(token_ids)}

    all_rows: list[dict] = []

    for h in range(n_heads):
        att = arr[h]
        L = len(token_ids)

        # Build graph from thresholded attention
        A = att > threshold
        np.fill_diagonal(A, False)
        if undirected:
            M = np.logical_or(A, A.T)
            G = nx.Graph()
            G.add_nodes_from(token_ids)
            tri = np.argwhere(np.triu(M, 1))
            for s, t in tri:
                w = 0.5 * (float(att[s, t]) + float(att[t, s]))
                G.add_edge(token_ids[s], token_ids[t], weight=w)
        else:
            G = nx.DiGraph()
            G.add_nodes_from(token_ids)
            idx = np.argwhere(A)
            for s, t in idx:
                w = float(att[s, t])
                G.add_edge(token_ids[s], token_ids[t], weight=w)

        # Communities
        if G.number_of_edges() > 0 and G.number_of_nodes() > 0:
            comms = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
        else:
            comms = [set(G.nodes())] if G.number_of_nodes() > 0 else []
        num_clusters = max(1, len(comms)) if G.number_of_nodes() > 0 else 0
        node_to_comm: dict[Any, int] = {}
        for i, cset in enumerate(comms):
            for n in cset:
                node_to_comm[n] = i

        # Colors per community (per head)
        cmap = cm.get_cmap(cmap_name, max(1, num_clusters))

        # Emit one row per token in original order
        for position, tid in enumerate(token_ids):
            comm_id = node_to_comm.get(tid, 0)
            color_hex = _rgba_to_hex(cmap(comm_id))
            gene_name = _resolve_gene_name(tid, position, pos_to_name, id_to_name)
            all_rows.append(
                {
                    "record_idx": record_idx,
                    "layer": layer_idx,
                    "head": h + 1,  # 1-based for readability
                    "position": position,
                    "token_id": tid,
                    "gene_name": gene_name,
                    "community": int(comm_id),
                    "color_hex": color_hex,
                }
            )

    df = pd.DataFrame(all_rows)
    meta = {
        "token_ids": token_ids,
        "gene_names": [
            _resolve_gene_name(tid, i, pos_to_name, id_to_name) for i, tid in enumerate(token_ids)
        ],
        "num_heads": int(n_heads),
        "num_tokens": int(len(token_ids)),
    }
    return df, meta


def plot_gene_community_heatmap(
    table: pd.DataFrame,
    title: str | None = None,
    out_path: str | None = None,
    show_gene_labels: bool = False,
    label_every: int = 10,
    background_color: str = "#ffffff",
    figsize: tuple[float, float] | None = None,
):
    """
    Plot a simple color-strip heatmap of gene-to-community assignment per head.

    Parameters
    ----------
    table : DataFrame
        Output of build_gene_community_table.
    title : str, optional
        Figure title.
    out_path : str, optional
        If provided, saves the figure to this path.
    show_gene_labels : bool
        If True, shows gene labels along x-axis (sparse).
    label_every : int
        Show every N-th gene label when show_gene_labels is True.
    background_color : str
        Hex color for cells with missing data.
    figsize : (w, h)
        Matplotlib figure size.
    """
    import matplotlib.pyplot as plt

    if table.empty:
        raise ValueError("'table' is empty. Build it first with build_gene_community_table().")

    heads = sorted(table["head"].unique())
    L = int(table["position"].max()) + 1

    # Prepare image array (H x L x 3)
    bg_rgb = _hex_to_rgb01(background_color)
    img = np.zeros((len(heads), L, 3), dtype=float)
    img[:] = bg_rgb

    # Fill colors per head/position
    for h_idx, head in enumerate(heads):
        sub = table[table["head"] == head]
        for _, row in sub.iterrows():
            pos = int(row["position"])  # 0-based
            rgb = _hex_to_rgb01(row["color_hex"])
            if 0 <= pos < L:
                img[h_idx, pos, :] = rgb

    # Auto figsize
    if figsize is None:
        w = max(8.0, min(24.0, L * 0.15))
        h = max(2.5, min(20.0, len(heads) * 0.6))
        figsize = (w, h)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img, aspect="auto")
    ax.set_yticks(range(len(heads)))
    ax.set_yticklabels([f"Head {h}" for h in heads])

    if show_gene_labels:
        # Use labels from the smallest head index present at each position
        template = (
            table.sort_values(["position", "head"])  # prefer lower head index for label
            .drop_duplicates(subset=["position"])  # one label per position
            .sort_values("position")
        )
        positions = template["position"].astype(int).tolist()
        labels = template["gene_name"].astype(str).tolist()
        tick_positions = positions[:: max(1, label_every)]
        tick_labels = labels[:: max(1, label_every)]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
    else:
        ax.set_xticks([])

    ax.set_xlabel("Gene position in sequence")
    ax.set_ylabel("Attention head")
    if title:
        ax.set_title(title)
    ax.grid(False)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()



# %%
df_tbl, meta = build_gene_community_table(rec, record_idx=0, layer_idx=0, threshold=0.0275, max_heads=6, undirected=True, cmap_name="tab20")
plot_gene_community_heatmap(df_tbl, title="Gene communities per head", out_path=None, show_gene_labels=True, label_every=5)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from networkx.algorithms import community as nx_comm
from typing import Any


def _compute_communities_from_attention(
    att: np.ndarray,
    token_ids: list,
    threshold: float,
    undirected: bool = True,
    cmap_name: str = "tab20",
):
    """
    Compute communities from an attention matrix via thresholded graph and return per-position colors.

    Returns (community_index_per_token, color_hex_per_token, num_clusters)
    """
    L = len(token_ids)
    A = att >= threshold
    np.fill_diagonal(A, False)

    if undirected:
        M = np.logical_or(A, A.T)
        G = nx.Graph()
        G.add_nodes_from(token_ids)
        tri = np.argwhere(np.triu(M, 1))
        for s, t in tri:
            w = 0.5 * (float(att[s, t]) + float(att[t, s]))
            G.add_edge(token_ids[s], token_ids[t], weight=w)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(token_ids)
        idx = np.argwhere(A)
        for s, t in idx:
            w = float(att[s, t])
            G.add_edge(token_ids[s], token_ids[t], weight=w)

    if G.number_of_edges() > 0 and G.number_of_nodes() > 0:
        comms = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
    else:
        comms = [set(G.nodes())] if G.number_of_nodes() > 0 else []

    num_clusters = max(1, len(comms)) if G.number_of_nodes() > 0 else 0
    node_to_comm: dict[Any, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node_to_comm[n] = i

    cmap = cm.get_cmap(cmap_name, max(1, num_clusters))
    labels = [node_to_comm.get(tid, 0) for tid in token_ids]
    colors = [
        f"#{int(round(255*r)):02x}{int(round(255*g)):02x}{int(round(255*b)):02x}"
        for (r, g, b, _a) in map(cmap, labels)
    ]
    return labels, colors, num_clusters


def clustermap_attention_with_community_colors(
    rec,
    record_idx: int = 0,
    layer_idx: int = 0,
    head_idx: int = 0,
    threshold_base: float = 0.0275,
    threshold_strict: float | None = 0.1,
    undirected: bool = True,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    heatmap_cmap: str = "Spectral",
    metric: str = "euclidean",
    method: str = "average",
    z_score: int | None = None,
    standard_scale: int | None = None,
    place_x_ticks_top: bool = True,
    figsize: tuple[float, float] | None = None,
    out_path: str | None = None,
):
    """
    Cluster the attention matrix (rows and cols) and color both axes by network communities
    computed at two edge thresholds (base and strict).

    - threshold_base: used to build graph -> communities (color bar 1)
    - threshold_strict: if provided, build another graph with stricter threshold (color bar 2)
    - Colors are stacked as two bars for rows and columns.
    """
    # Token ids
    ids = rec[record_idx]["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    token_ids = list(ids[0]) if isinstance(ids, list) and ids and isinstance(ids[0], (list, np.ndarray)) else list(ids)

    # Attention for layer/head
    heads = rec[record_idx]["attentions"][layer_idx]
    arr = heads.detach().cpu().numpy() if hasattr(heads, "detach") else np.array(heads)
    if arr.ndim == 2:
        arr = arr[None, ...]
    att = arr[head_idx]

    # Prepare matrix
    M = 0.5 * (att + att.T) if symmetric else att.copy()
    if zero_diagonal:
        np.fill_diagonal(M, 0.0)

    # Community colors at base threshold
    base_labels, base_colors, _ = _compute_communities_from_attention(
        att, token_ids, threshold=threshold_base, undirected=undirected, cmap_name="tab20"
    )

    # Optional strict threshold
    if threshold_strict is not None:
        strict_labels, strict_colors, _ = _compute_communities_from_attention(
            att, token_ids, threshold=threshold_strict, undirected=undirected, cmap_name="Set2"
        )
        row_colors = pd.DataFrame({"base": base_colors, "strict": strict_colors})
        col_colors = pd.DataFrame({"base": base_colors, "strict": strict_colors})
    else:
        row_colors = pd.DataFrame({"base": base_colors})
        col_colors = pd.DataFrame({"base": base_colors})

    # Auto figsize
    if figsize is None:
        L = len(token_ids)
        w = max(8.0, min(22.0, 0.15 * L))
        h = max(8.0, min(22.0, 0.15 * L))
        figsize = (w, h)

    # Draw clustermap
    g = sns.clustermap(
        M,
        cmap=heatmap_cmap,
        method=method,
        metric=metric,
        z_score=z_score,
        standard_scale=standard_scale,
        row_colors=row_colors,
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=False,
        figsize=figsize,
        cbar_kws={"label": "attention"},
    )

    if place_x_ticks_top:
        g.ax_heatmap.xaxis.set_ticks_position("top")
        g.ax_heatmap.xaxis.set_label_position("top")
        g.ax_heatmap.tick_params(top=True, bottom=False)

    g.fig.suptitle(
        f"Layer {layer_idx} – Head {head_idx+1} clustermap | base thr={threshold_base}"
        + (f", strict thr={threshold_strict}" if threshold_strict is not None else ""),
        y=1.02,
    )
    plt.tight_layout()
    if out_path:
        g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "row_colors": row_colors,
        "col_colors": col_colors,
        "row_order": g.dendrogram_row.reordered_ind if g.dendrogram_row is not None else None,
        "col_order": g.dendrogram_col.reordered_ind if g.dendrogram_col is not None else None,
    }


def clustermap_all_heads(
    rec,
    record_idx: int = 0,
    layer_idx: int = 0,
    threshold_base: float = 0.0275,
    threshold_strict: float | None = 0.1,
    undirected: bool = True,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    heatmap_cmap: str = "Spectral",
    metric: str = "euclidean",
    method: str = "average",
    max_heads: int = 6,
    out_prefix: str | None = None,
):
    heads = rec[record_idx]["attentions"][layer_idx]
    arr = heads.detach().cpu().numpy() if hasattr(heads, "detach") else np.array(heads)
    if arr.ndim == 2:
        arr = arr[None, ...]
    n_heads = min(max_heads, arr.shape[0])

    for h in range(n_heads):
        out_path = None
        if out_prefix:
            out_path = f"{out_prefix}_layer{layer_idx}_head{h+1}_clustermap.png"
        clustermap_attention_with_community_colors(
            rec,
            record_idx=record_idx,
            layer_idx=layer_idx,
            head_idx=h,
            threshold_base=threshold_base,
            threshold_strict=threshold_strict,
            undirected=undirected,
            symmetric=symmetric,
            zero_diagonal=zero_diagonal,
            heatmap_cmap=heatmap_cmap,
            metric=metric,
            method=method,
            figsize=None,
            out_path=out_path,
        )



# %%
clustermap_all_heads(rec, record_idx=0, layer_idx=0, threshold_base=0.0275, threshold_strict=0.1, out_prefix=None)

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def clustermap_attention_with_community_colors(
    rec,
    record_idx: int = 0,
    layer_idx: int = 0,
    head_idx: int = 0,
    threshold_base: float = 0.0275,
    threshold_strict: float | None = 0.1,
    undirected: bool = True,
    symmetric: bool = True,
    zero_diagonal: bool = True,
    heatmap_cmap: str = "Spectral",
    metric: str = "euclidean",
    method: str = "average",
    z_score: int | None = None,
    standard_scale: int | None = None,
    place_x_ticks_top: bool = True,
    figsize: tuple[float, float] | None = None,
    out_path: str | None = None,
):
    """
    Updated: pass row/col colors as non-indexed arrays to avoid seaborn index-matching errors.
    """
    # Token ids
    ids = rec[record_idx]["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    token_ids = list(ids[0]) if isinstance(ids, list) and ids and isinstance(ids[0], (list, np.ndarray)) else list(ids)

    # Attention for layer/head
    heads = rec[record_idx]["attentions"][layer_idx]
    arr = heads.detach().cpu().numpy() if hasattr(heads, "detach") else np.array(heads)
    if arr.ndim == 2:
        arr = arr[None, ...]
    att = arr[head_idx]

    # Prepare matrix
    M = 0.5 * (att + att.T) if symmetric else att.copy()
    if zero_diagonal:
        np.fill_diagonal(M, 0.0)

    # Community colors at base and optional strict threshold
    _, base_colors, _ = _compute_communities_from_attention(
        att, token_ids, threshold=threshold_base, undirected=undirected, cmap_name="tab20"
    )

    if threshold_strict is not None:
        _, strict_colors, _ = _compute_communities_from_attention(
            att, token_ids, threshold=threshold_strict, undirected=undirected, cmap_name="Set2"
        )
        # Provide non-indexed list-of-lists for multiple color bands
        row_colors = [base_colors, strict_colors]
        col_colors = [base_colors, strict_colors]
    else:
        # Provide a single non-indexed list for one color band
        row_colors = base_colors
        col_colors = base_colors

    # Auto figsize
    if figsize is None:
        L = len(token_ids)
        w = max(8.0, min(22.0, 0.15 * L))
        h = max(8.0, min(22.0, 0.15 * L))
        figsize = (w, h)

    # Draw clustermap
    g = sns.clustermap(
        M,
        cmap=heatmap_cmap,
        method=method,
        metric=metric,
        z_score=z_score,
        standard_scale=standard_scale,
        row_colors=row_colors,
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=False,
        figsize=figsize,
        cbar_kws={"label": "attention"},
    )

    if place_x_ticks_top:
        g.ax_heatmap.xaxis.set_ticks_position("top")
        g.ax_heatmap.xaxis.set_label_position("top")
        g.ax_heatmap.tick_params(top=True, bottom=False)

    g.fig.suptitle(
        f"Layer {layer_idx} – Head {head_idx+1} clustermap | base thr={threshold_base}"
        + (f", strict thr={threshold_strict}" if threshold_strict is not None else ""),
        y=1.02,
    )
    plt.tight_layout()
    if out_path:
        g.fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "row_colors": row_colors,
        "col_colors": col_colors,
        "row_order": g.dendrogram_row.reordered_ind if g.dendrogram_row is not None else None,
        "col_order": g.dendrogram_col.reordered_ind if g.dendrogram_col is not None else None,
    }



# %%
clustermap_all_heads(rec, record_idx=0, layer_idx=0, threshold_base=0.0275, threshold_strict=0.1)


