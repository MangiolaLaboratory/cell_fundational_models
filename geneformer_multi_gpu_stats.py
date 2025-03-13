

import os
import json
import time
import pickle
import datetime
import numpy as np
from datetime import timedelta

import torch
import scanpy as sc
import datasets
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer, BertForSequenceClassification
from geneformer import DataCollatorForCellClassification, EmbExtractor

def get_gpu_memory():
    """Get current GPU memory usage"""
    gpu_memory = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.memory_allocated(i) / (1024**3)  #GB
        gpu_memory.append(mem)
    return gpu_memory

def track_gpu_usage(func):
    def wrapper(*args, **kwargs):
        #  memory
        start_mem = get_gpu_memory()
        #start time
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        duration = timedelta(seconds=end_time - start_time)
        end_mem = get_gpu_memory()
        # Calculate time
        used_mem = [end - start for start, end in zip(start_mem, end_mem)]
        print(f"Function {func.__name__} completed in {duration}")
        for i, mem in enumerate(used_mem):
            print(f"GPU {i} memory used: {mem:.2f} GB")
        return result
    return wrapper

def log_training_stats(trainer, output_dir="./logs"):
    """Extract and plot training statistics"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get training logs
    logs = trainer.state.log_history
    
    # Extract training and validation losses
    train_losses = []
    train_steps = []
    eval_losses = []
    eval_steps = []
    
    for entry in logs:
        if 'loss' in entry:
            train_losses.append(entry['loss'])
            train_steps.append(entry['step'])
        elif 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry['step'])
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, 'b-', label='Training Loss')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/loss_plot.png")
    plt.close()
    
    # Compute summary statistics
    stats = {
        "gpu_count": torch.cuda.device_count(),
        "peak_gpu_memory_gb": [torch.cuda.max_memory_allocated(i) / (1024**3) for i in range(torch.cuda.device_count())],
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "train_steps": len(train_losses)
    }
    
    # Print summary
    print("\n===== TRAINING SUMMARY =====")
    print(f"GPU Count: {stats['gpu_count']}")
    for i, mem in enumerate(stats['peak_gpu_memory_gb']):
        print(f"GPU {i} Peak Memory: {mem:.2f} GB")
    if stats['final_train_loss']:
        print(f"Final Training Loss: {stats['final_train_loss']:.4f}")
    if stats['final_eval_loss']:
        print(f"Final Validation Loss: {stats['final_eval_loss']:.4f}")
    print(f"Training Steps: {stats['train_steps']}")
    print("============================\n")
    
    return stats

def log_gpu_scaling(gpu_count, batch_size, training_time, num_samples):
    """Log GPU scaling metrics"""
    throughput = num_samples / training_time
    print("\n===== GPU SCALING METRICS =====")
    print(f"Number of GPUs: {gpu_count}")
    print(f"Batch size per GPU: {batch_size}")
    print(f"Total training time: {timedelta(seconds=training_time)}")
    print(f"Throughput: {throughput:.2f} samples/second")
    print(f"Throughput per GPU: {throughput/gpu_count:.2f} samples/second/GPU")
    print("==============================\n")


def main():
    os.environ["WANDB_MODE"] = "disabled"
    
    #paths
    base_dir = "/hpcfs/users/a1841503/Geneformer"
    model_dir = f"{base_dir}/fine_tuned_geneformer"
    token_dir = f"{base_dir}/cellxgene_version_long/data/tokenized_data/"
    output_dir = f"{base_dir}/fine_tuned_models/CellNexus"
    figures_dir = f"{base_dir}/cellxgene_version_long/figures"
    embedding_dir = f"{base_dir}/cellxgene_version_long/data/geneformer_embeddings/"
    
    # Create output directories if they don't exist
    for directory in [output_dir, figures_dir, embedding_dir, "./results", "./logs", "./performance_logs"]:
        os.makedirs(directory, exist_ok=True)
    
    # Load label mapping dictionary
    label_mapping_dict_file = os.path.join(model_dir, "label_to_cell_subclass.json")
    with open(label_mapping_dict_file) as fp:
        label_mapping_dict = json.load(fp)
    
    # Load and prepare dataset
    dataset = datasets.load_from_disk(token_dir + "m_f_20_20.dataset")
    dataset = dataset.add_column("label", [0] * len(dataset))
    
    # Load token dictionary
    with open(f'{base_dir}/geneformer/gene_dictionaries_30m/token_dictionary_gc30M.pkl', 'rb') as f:
        token_dict = pickle.load(f)
    
    # Split data into train and test sets
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset_dict['train']
    eval_dataset = dataset_dict['test']
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=32,        
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.02,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-5,
        gradient_checkpointing=False,
    )
    
    # Load pre-trained model 
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=88,
        hidden_dropout_prob=0.02,
        attention_probs_dropout_prob=0.02,
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_hidden_layers=12,
        problem_type="single_label_classification"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForCellClassification(token_dictionary=token_dict),
    )
    
    # Log start time
    current_date = datetime.datetime.now()
    datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
    print(f"Training started at: {datestamp}")
    start_time = time.monotonic()
    
    # Train the model
    @track_gpu_usage
    def run_training(trainer):
        return trainer.train()
    
    train_results = run_training(trainer)
    
    # Log performance after training
    training_logs = trainer.state.log_history
    end_time = time.monotonic()
    training_time = end_time - start_time
    
    # Log training completion time
    after_training_date = datetime.datetime.now()
    datestamp_after_training = f"{str(after_training_date.year)[-2:]}{after_training_date.month:02d}{after_training_date.day:02d}{after_training_date.hour:02d}{after_training_date.minute:02d}{after_training_date.second:02d}"
    print(f"Training completed at: {datestamp_after_training}")
    print(f"Total training time: {timedelta(seconds=training_time)}")
    
    # Log GPU scaling metrics
    log_gpu_scaling(
        gpu_count=torch.cuda.device_count(),
        batch_size=training_args.per_device_train_batch_size,
        training_time=training_time,
        num_samples=len(train_dataset)
    )
    
    # Log detailed training stats
    log_training_stats(trainer, output_dir="./performance_logs")
    
    # Save the trained model
    trainer.save_model(output_dir)
    
    # Evaluate the model
    start_time = time.monotonic()
    eval_results = trainer.evaluate()
    end_time = time.monotonic()
    eval_time = end_time - start_time
    
    print(f"Evaluation time: {timedelta(seconds=eval_time)}")
    print(f"Evaluation results: {eval_results}")
    
    # Extract embeddings and visualize
    process_embeddings_and_visualize(
        model_dir=model_dir,
        token_dir=token_dir,
        embedding_dir=embedding_dir,
        figures_dir=figures_dir,
        label_mapping_dict=label_mapping_dict,
        training_logs=training_logs
    )


def process_embeddings_and_visualize(model_dir, token_dir, embedding_dir, figures_dir, label_mapping_dict, training_logs):
    """Process embeddings and generate visualizations"""
    
    # Load anndata object
    adata = sc.read_h5ad("/hpcfs/users/a1841503/Geneformer/cellxgene_version_long/data/h5ad/m_f_20_20.h5ad")
    adata_geneformer = adata.copy()
    
    # Make sure embedding directory exists
    os.makedirs(embedding_dir, exist_ok=True)
    
    # Set up embedding extractor
    n_classes = len(label_mapping_dict)
    embex = EmbExtractor(
        model_type="CellClassifier",
        num_classes=n_classes,
        emb_mode="cell",
        max_ncells=None,
        emb_label=["joinid"],
        emb_layer=0,
        forward_batch_size=30,
        nproc=8,
    )
    
    # Extract embeddings
    embs = embex.extract_embs(
        model_directory=model_dir,
        input_data_file=token_dir + "m_f_20_20.dataset",
        output_directory=embedding_dir,
        output_prefix="emb_long_2_GPU",
    )
    
    # Sort embeddings and add to anndata object
    embs = embs.sort_values("joinid")
    adata_geneformer.obsm["geneformer"] = embs.drop(columns="joinid").to_numpy()
    
    # Extract and plot training/evaluation losses
    plot_training_losses(training_logs, figures_dir)
    
    # Process data for UMAP visualization
    adata,adata_geneformer=prepare_data_for_umap(adata, adata_geneformer)

    # Generate UMAP visualizations
    generate_umap_visualizations(adata, adata_geneformer, figures_dir)


def plot_training_losses(training_logs, figures_dir):
    """Plot training and evaluation losses"""
    
    # Extract training and evaluation losses
    train_losses = [log["loss"] for log in training_logs if "loss" in log]
    train_steps = [log["step"] for log in training_logs if "loss" in log]
    
    eval_losses = [log["eval_loss"] for log in training_logs if "eval_loss" in log]
    eval_steps = [log["step"] for log in training_logs if "eval_loss" in log]
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_losses, 'b-', label='Training Loss')
    plt.plot(eval_steps, eval_losses, 'r-', label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_2.png')
    plt.savefig(f'{figures_dir}/loss_2.png')
    plt.close()


def prepare_data_for_umap(adata, adata_geneformer):
    """Prepare anndata objects for UMAP visualization"""
    
    # Process original data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    
    # Process Geneformer embeddings
    sc.pp.neighbors(adata_geneformer, n_neighbors=10, n_pcs=40, use_rep="geneformer")
    sc.tl.umap(adata_geneformer)
    return adata,adata_geneformer

def generate_umap_visualizations(adata, adata_geneformer, figures_dir):
    """Generate UMAP visualizations comparing original data with Geneformer embeddings"""
    
    # Set figure parameters
    sc.set_figure_params(figsize=(15, 5))
    
    # List of features to visualize
    features = [
        {"name": "sample_id", "legend_loc": "none"},
        {"name": "dataset_id", "legend_loc": "best"},
        {"name": "cell_type", "legend_loc": "none"},
        {"name": "sex", "legend_loc": "best"},
        {"name": "age_days", "legend_loc": "best"},
        {"name": "self_reported_ethnicity", "legend_loc": "lower left"}
    ]
    
    # Generate UMAP visualizations for each feature
    for feature in features:
        feature_name = feature["name"]
        legend_loc = feature["legend_loc"]
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        # Plot original data
        sc.pl.umap(
            adata, 
            color=feature_name, 
            title="Original Annotations",
            show=False, 
            ax=ax1,
            legend_loc=legend_loc
        )
        
        # Plot Geneformer embeddings
        sc.pl.umap(
            adata_geneformer, 
            color=feature_name, 
            title="Geneformer Annotations",
            show=False, 
            ax=ax2,
            legend_loc=legend_loc
        )
        
        # Save figure with correct filename
        plt.savefig(f'{feature_name}_2.png')
        plt.savefig(f'{figures_dir}/{feature_name}_2.png')
        plt.close(fig)


if __name__ == "__main__":
    main()
