import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt
import matplotlib
# Set matplotlib backend to handle potential display issues
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend for saving plots
import numpy as np
from datetime import datetime
import json
import pickle

from utils import (
    BilingualDataset, causal_mask, load_custom_dataset, get_or_build_tokenizer,
    InputEmbedding, PositionalEncoding, RotaryPositionalEmbedding, RelativePositionalBias, ProjectionLayer,
    MultiHeadAttention, feedForwardBlock, greedy_decode
)
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock
from config import get_config, get_weights_file_path

try:
    import torchmetrics
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("torchmetrics not available. Some metrics will be skipped.")


class MetricsTracker:
    """Class to track and plot training metrics"""
    
    def __init__(self, save_dir="plots"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.train_losses = []
        self.validation_bleu_scores = []
        self.epochs = []
        self.global_steps = []
        
        # For loss tracking within epochs
        self.batch_losses = []
        self.batch_steps = []
        
    def add_train_loss(self, loss, global_step):
        """Add a training loss value"""
        self.batch_losses.append(loss)
        self.batch_steps.append(global_step)
        
    def add_epoch_metrics(self, epoch, bleu_score):
        """Add end-of-epoch metrics"""
        self.epochs.append(epoch)
        self.validation_bleu_scores.append(bleu_score)
        
        # Calculate average loss for this epoch
        if self.batch_losses:
            epoch_avg_loss = np.mean(self.batch_losses[-len(self.batch_losses)//max(1, len(self.epochs)):])
            self.train_losses.append(epoch_avg_loss)
        
    def plot_metrics(self, save_plots=True):
        """Create and save plots of training metrics"""
        
        # Set up the plot style
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Training Loss over Epochs
        if self.train_losses and self.epochs:
            ax1.plot(self.epochs, self.train_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Average Loss', fontsize=12)
            ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add value annotations on points
            for i, (epoch, loss) in enumerate(zip(self.epochs, self.train_losses)):
                if i % max(1, len(self.epochs)//10) == 0:  # Annotate every 10th point to avoid clutter
                    ax1.annotate(f'{loss:.3f}', (epoch, loss), 
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        # Plot 2: BLEU Score over Epochs
        if self.validation_bleu_scores and self.epochs:
            ax2.plot(self.epochs, self.validation_bleu_scores, 'g-o', linewidth=2, markersize=6, label='Validation BLEU')
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('BLEU Score', fontsize=12)
            ax2.set_title('Validation BLEU Score over Epochs', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add value annotations on points
            for i, (epoch, bleu) in enumerate(zip(self.epochs, self.validation_bleu_scores)):
                if i % max(1, len(self.epochs)//10) == 0:  # Annotate every 10th point
                    ax2.annotate(f'{bleu:.4f}', (epoch, bleu), 
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_plots:
            # Save the plot with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = self.save_dir / f"training_metrics_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {plot_filename}")
            
            # Also save without timestamp for latest version
            latest_filename = self.save_dir / "training_metrics_latest.png"
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
            print(f"Latest training metrics plot saved to: {latest_filename}")
        
        # Don't show plot in headless environment, just close it
        plt.close()
        
    def plot_detailed_loss(self, save_plots=True):
        """Create detailed loss plot showing batch-level loss"""
        if not self.batch_losses:
            print("No batch loss data available for detailed plot")
            return
            
        plt.figure(figsize=(15, 6))
        plt.plot(self.batch_steps, self.batch_losses, 'b-', alpha=0.7, linewidth=1, label='Batch Loss')
        
        # Add smoothed line
        if len(self.batch_losses) > 10:
            window_size = max(10, len(self.batch_losses) // 50)
            smoothed_loss = np.convolve(self.batch_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = self.batch_steps[window_size-1:]
            plt.plot(smoothed_steps, smoothed_loss, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        plt.xlabel('Global Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Detailed Training Loss over Global Steps', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = self.save_dir / f"detailed_loss_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Detailed loss plot saved to: {plot_filename}")
            
            latest_filename = self.save_dir / "detailed_loss_latest.png"
            plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
        
        # Don't show plot in headless environment, just close it
        plt.close()
        
    def save_metrics_data(self):
        """Save metrics data to files for later analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary data
        summary_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'validation_bleu_scores': self.validation_bleu_scores,
            'timestamp': timestamp
        }
        
        import json
        summary_file = self.save_dir / f"metrics_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Metrics summary saved to: {summary_file}")
        
        # Save detailed batch data
        detailed_data = {
            'batch_steps': self.batch_steps,
            'batch_losses': self.batch_losses,
            'timestamp': timestamp
        }
        
        detailed_file = self.save_dir / f"detailed_metrics_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_data, f, indent=2)
        print(f"Detailed metrics saved to: {detailed_file}")


def save_dataset_partitions(train_ds_raw, val_ds_raw, test_ds_raw, config, save_dir="dataset_splits"):
    """
    Save the dataset partitions to files for reproducibility and analysis
    
    Args:
        train_ds_raw: Raw training dataset partition
        val_ds_raw: Raw validation dataset partition  
        test_ds_raw: Raw test dataset partition
        config: Configuration dictionary
        save_dir: Directory to save the partitions
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nSaving dataset partitions to '{save_dir}' directory...")
    
    # Save partition metadata
    partition_info = {
        'timestamp': timestamp,
        'total_samples': len(train_ds_raw) + len(val_ds_raw) + len(test_ds_raw),
        'train_size': len(train_ds_raw),
        'val_size': len(val_ds_raw),
        'test_size': len(test_ds_raw),
        'train_ratio': len(train_ds_raw) / (len(train_ds_raw) + len(val_ds_raw) + len(test_ds_raw)),
        'val_ratio': len(val_ds_raw) / (len(train_ds_raw) + len(val_ds_raw) + len(test_ds_raw)),
        'test_ratio': len(test_ds_raw) / (len(train_ds_raw) + len(val_ds_raw) + len(test_ds_raw)),
        'config': {
            'lang_src': config.get('lang_src', 'unknown'),
            'lang_tgt': config.get('lang_tgt', 'unknown'),
            'seq_len': config.get('seq_len', 'unknown'),
            'batch_size': config.get('batch_size', 'unknown')
        }
    }
    
    # Save metadata as JSON
    metadata_file = save_path / f"partition_info_{timestamp}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(partition_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Partition metadata saved to: {metadata_file}")
    
    # Save latest metadata (without timestamp for easy access)
    latest_metadata_file = save_path / "partition_info_latest.json"
    with open(latest_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(partition_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Latest partition metadata saved to: {latest_metadata_file}")
    
    # Save partition indices (which samples went to which split)
    # This is useful for reproducibility
    partition_indices = {
        'train_indices': train_ds_raw.indices if hasattr(train_ds_raw, 'indices') else list(range(len(train_ds_raw))),
        'val_indices': val_ds_raw.indices if hasattr(val_ds_raw, 'indices') else list(range(len(val_ds_raw))),
        'test_indices': test_ds_raw.indices if hasattr(test_ds_raw, 'indices') else list(range(len(test_ds_raw))),
        'timestamp': timestamp
    }
    
    indices_file = save_path / f"partition_indices_{timestamp}.json"
    with open(indices_file, 'w') as f:
        json.dump(partition_indices, f, indent=2)
    print(f"✅ Partition indices saved to: {indices_file}")
    
    # Save latest indices
    latest_indices_file = save_path / "partition_indices_latest.json"
    with open(latest_indices_file, 'w') as f:
        json.dump(partition_indices, f, indent=2)
    print(f"✅ Latest partition indices saved to: {latest_indices_file}")
    
    # Save actual data samples in text format for easy inspection
    def save_partition_samples(partition_data, partition_name, sample_limit=100):
        """Save a subset of samples from each partition in readable text format"""
        samples_file = save_path / f"{partition_name}_samples_{timestamp}.txt"
        latest_samples_file = save_path / f"{partition_name}_samples_latest.txt"
        
        with open(samples_file, 'w', encoding='utf-8') as f, \
             open(latest_samples_file, 'w', encoding='utf-8') as f_latest:
            
            # Write header
            header = f"=== {partition_name.upper()} PARTITION SAMPLES ===\n"
            header += f"Total samples in partition: {len(partition_data)}\n"
            header += f"Showing first {min(sample_limit, len(partition_data))} samples\n"
            header += f"Timestamp: {timestamp}\n"
            header += "=" * 50 + "\n\n"
            
            f.write(header)
            f_latest.write(header)
            
            # Write samples
            for i, item in enumerate(partition_data):
                if i >= sample_limit:
                    break
                    
                # Extract the translation data
                if hasattr(item, 'translation'):
                    translation_data = item.translation
                elif isinstance(item, dict) and 'translation' in item:
                    translation_data = item['translation']
                else:
                    translation_data = item
                
                sample_text = f"Sample {i+1}:\n"
                if isinstance(translation_data, dict):
                    for lang, text in translation_data.items():
                        sample_text += f"  {lang}: {text}\n"
                else:
                    sample_text += f"  Data: {translation_data}\n"
                sample_text += "\n"
                
                f.write(sample_text)
                f_latest.write(sample_text)
        
        print(f"✅ {partition_name.capitalize()} samples saved to: {samples_file}")
        return samples_file
    
    # Save sample data from each partition
    try:
        save_partition_samples(train_ds_raw, "train", sample_limit=50)
        save_partition_samples(val_ds_raw, "validation", sample_limit=25)
        save_partition_samples(test_ds_raw, "test", sample_limit=25)
    except Exception as e:
        print(f"⚠️  Warning: Could not save sample data: {e}")
    
    # Save partition objects using pickle for complete reproducibility
    try:
        partitions_pickle_file = save_path / f"partitions_complete_{timestamp}.pkl"
        with open(partitions_pickle_file, 'wb') as f:
            pickle.dump({
                'train_ds_raw': train_ds_raw,
                'val_ds_raw': val_ds_raw,
                'test_ds_raw': test_ds_raw,
                'metadata': partition_info
            }, f)
        print(f"✅ Complete partition objects saved to: {partitions_pickle_file}")
        
        # Save latest pickle file
        latest_pickle_file = save_path / "partitions_complete_latest.pkl"
        with open(latest_pickle_file, 'wb') as f:
            pickle.dump({
                'train_ds_raw': train_ds_raw,
                'val_ds_raw': val_ds_raw,
                'test_ds_raw': test_ds_raw,
                'metadata': partition_info
            }, f)
        print(f"✅ Latest complete partition objects saved to: {latest_pickle_file}")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not save partition objects with pickle: {e}")
    
    print(f"\n📁 Dataset partition saving completed!")
    print(f"   - Metadata and indices saved as JSON files")
    print(f"   - Sample data saved as text files for inspection")
    print(f"   - Complete partition objects saved as pickle files")
    print(f"   - All files available in '{save_dir}' directory")
    
    return save_path


# Optional imports


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection = proj
        
    def encode(self, src, src_mask):
        # Cast source tensor to long, as embedding layers expect integer indices
        src = self.src_embed(src.long())
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # Cast target tensor to long, as embedding layers expect integer indices
        tgt = self.tgt_embed(tgt.long())
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=6, d_ff=1024, h=8, dropout=0.1, use_rope=None, positional_encoding=None, relative_bias_clamp_distance=32):
    # Create the components of the transformer model
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)
    
    # Determine positional encoding type
    if positional_encoding is None:
        # Fallback to legacy use_rope parameter for backward compatibility
        positional_encoding = "rope" if use_rope else "vanilla"
    
    # Choose positional encoding type based on configuration
    if positional_encoding == "rope":
        print(f"Using Rotary Positional Embeddings (RoPE)")
        # For RoPE, we use head dimension (d_k = d_model // h) instead of full d_model
        d_k = d_model // h
        src_pos = RotaryPositionalEmbedding(d_k, src_seq_len, dropout)
        tgt_pos = RotaryPositionalEmbedding(d_k, tgt_seq_len, dropout)
        # For RoPE, we pass the rope instance to attention layers
        src_rope = src_pos
        tgt_rope = tgt_pos
        src_relative_bias = None
        tgt_relative_bias = None
        
    elif positional_encoding == "relative_bias":
        print(f"Using Relative Position Bias")
        # For relative bias, we don't need traditional positional encodings
        # Create dummy positional encodings that just apply dropout
        class DummyPositionalEncoding(nn.Module):
            def __init__(self, dropout):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                return self.dropout(x)
        
        src_pos = DummyPositionalEncoding(dropout)
        tgt_pos = DummyPositionalEncoding(dropout)
        # Create relative bias instances
        src_relative_bias = RelativePositionalBias(h, src_seq_len, dropout, relative_bias_clamp_distance)
        tgt_relative_bias = RelativePositionalBias(h, tgt_seq_len, dropout, relative_bias_clamp_distance)
        # No rope for relative bias
        src_rope = None
        tgt_rope = None
        
    else:  # vanilla
        print(f"Using vanilla positional encodings")
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        # For vanilla, no rope or relative bias
        src_rope = None
        tgt_rope = None
        src_relative_bias = None
        tgt_relative_bias = None
    
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout, rope=src_rope, relative_bias=src_relative_bias)
        feed_forward_block = feedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout, rope=tgt_rope, relative_bias=tgt_relative_bias)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout, rope=None, relative_bias=None)  # Cross-attention doesn't use positional encoding between different sequences
        feed_forward_block = feedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block) 

    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer


def get_ds(config):
    # Load custom dataset from local files
    fi_file_path = "datasets/EUbookshop.fi"
    en_file_path = "datasets/EUbookshop.en"
    
    # small_fi_file_path = "datasets/smallFI.fi"
    # small_en_file_path = "datasets/smallEN.en"
    
    ds_raw = load_custom_dataset(fi_file_path, en_file_path)
    
    # ds_raw = load_custom_dataset(small_fi_file_path, small_en_file_path, save_path="datasets/small/small_custom_dataset.json")
    
    # Original opus books dataset loading (commented out)
    # ds_raw = load_dataset('opus_books', lang1=config['lang_src'], lang2=config['lang_tgt'], split='train')
    
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    train_ds_size = int(0.8 * len(ds_raw))
    val_ds_size = int(0.1 * len(ds_raw))
    test_ds_size = len(ds_raw) - train_ds_size - val_ds_size

    train_ds_raw, val_ds_raw, test_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size, test_ds_size])
    
    # Save dataset partitions for reproducibility and analysis
    save_dataset_partitions(train_ds_raw, val_ds_raw, test_ds_raw, config)
    
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    test_ds = BilingualDataset(test_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len']) 
    
    # Calculate statistics on the original dataset (before filtering)
    max_len_src = 0
    max_len_tgt = 0
    samples_over_limit = 0
    
    print("Analyzing sequence lengths in original dataset...")
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids 

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
        # Check if this sample would exceed our sequence length limits
        if (len(src_ids) + 2 > config['seq_len']) or (len(tgt_ids) + 1 > config['seq_len']):
            samples_over_limit += 1

    print(f"Original dataset statistics:")
    print(f"  Max source length: {max_len_src}, Max target length: {max_len_tgt}")
    print(f"  Samples exceeding seq_len={config['seq_len']}: {samples_over_limit}/{len(ds_raw)} ({samples_over_limit/len(ds_raw)*100:.1f}%)")
    print(f"  Recommended seq_len for full coverage: {max(max_len_src + 2, max_len_tgt + 1)}")
    
    # Calculate total samples after filtering
    total_filtered_samples = len(train_ds) + len(val_ds) + len(test_ds)
    original_samples = len(ds_raw)
    retention_rate = total_filtered_samples / original_samples * 100
    
    print(f"\nDataset sizes after filtering:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Validation: {len(val_ds)} samples") 
    print(f"  Test: {len(test_ds)} samples")
    print(f"  Total retained: {total_filtered_samples}/{original_samples} ({retention_rate:.1f}%)")
    
    # Create data loaders
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)  # Validation requires batch_size=1
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)  # Test requires batch_size=1

    return train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    # Get positional encoding configuration
    positional_encoding = config.get('positional_encoding', None)
    relative_bias_clamp_distance = config.get('relative_bias_clamp_distance', 32)
    
    # Legacy support for use_rope parameter
    use_rope = config.get('use_rope', False) if positional_encoding is None else None
    
    model = build_transformer(
        vocab_src_len, vocab_tgt_len, 
        config['seq_len'], config['seq_len'], 
        config['d_model'], 
        N=2, d_ff=1024, h=2, 
        use_rope=use_rope,
        positional_encoding=positional_encoding,
        relative_bias_clamp_distance=relative_bias_clamp_distance
    )
    return model


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2, max_validation_samples=100):
    """
    Run validation on the validation dataset and calculate BLEU score
    Limited to max_validation_samples for efficiency
    """
    model.eval()
    count = 0
    processed_samples = 0
    
    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width (works on Windows)
        import shutil
        console_width = shutil.get_terminal_size().columns
    except:
        console_width = 80

    print_msg(f"Running validation on up to {max_validation_samples} samples...")
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            
            # Limit validation samples for efficiency
            if processed_samples >= max_validation_samples:
                print_msg(f"Reached validation limit of {max_validation_samples} samples")
                break
                
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # Check that the batch size is 1 (required for greedy decoding)
            actual_batch_size = encoder_input.size(0)
            if actual_batch_size != 1:
                raise ValueError(f"Validation requires batch_size=1 for greedy decoding, but got batch_size={actual_batch_size}. "
                               f"Make sure validation DataLoader uses batch_size=1")

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            processed_samples += 1
            
            # Show progress every 100 samples
            if processed_samples % 100 == 0:
                print_msg(f"Validation progress: {processed_samples}/{max_validation_samples}")
            
            # Print the source, target and model output
            if count <= num_examples:
                print_msg('-'*console_width)
                print_msg(f"{f'SOURCE: ':>12}{source_text}")
                print_msg(f"{f'TARGET: ':>12}{target_text}")
                print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

    if writer:
        if TORCHMETRICS_AVAILABLE:
            # Evaluate the character error rate
            # Compute the char error rate 
            metric = torchmetrics.CharErrorRate()
            cer = metric(predicted, expected)
            writer.add_scalar('validation cer', cer, global_step)
            writer.flush()

            # Compute the word error rate
            metric = torchmetrics.WordErrorRate()
            wer = metric(predicted, expected)
            writer.add_scalar('validation wer', wer, global_step)
            writer.flush()

            # Compute the BLEU metric
            metric = torchmetrics.BLEUScore()
            bleu = metric(predicted, expected)
            writer.add_scalar('validation BLEU', bleu, global_step)
            writer.flush()
        else:
            print_msg("torchmetrics not available, skipping CER, WER, and torchmetrics BLEU calculation")
        
    # Calculate BLEU score using NLTK
    print_msg(f"Calculating BLEU score on {len(expected)} validation samples...")
    try:
        # Convert expected and predicted to the format required by corpus_bleu
        # corpus_bleu expects a list of lists of reference sentences and a list of candidate sentences
        references = [[exp.split()] for exp in expected]  # List of lists of reference tokens
        candidates = [pred.split() for pred in predicted]  # List of candidate tokens
        
        bleu_score = corpus_bleu(references, candidates)
        print_msg(f'BLEU Score: {bleu_score:.4f}')
        
        if writer:
            writer.add_scalar('validation BLEU (NLTK)', bleu_score, global_step)
            writer.flush()
            
    except Exception as e:
        print_msg(f"Error calculating BLEU score: {e}")
        bleu_score = 0.0
    
    print_msg("Validation completed!")
    
    return bleu_score


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir="plots")
    
    # Multi-GPU setup
    num_gpus = torch.cuda.device_count()
    use_multi_gpu = num_gpus > 1
    
    if use_multi_gpu:
        print(f"Found {num_gpus} GPUs - enabling multi-GPU training")
        print(f"GPUs available:")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Adjust batch size for multiple GPUs
        effective_batch_size = config['batch_size'] * num_gpus
        print(f"Effective batch size with {num_gpus} GPUs: {effective_batch_size}")
    else:
        print("Single GPU or CPU training")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, test_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size())
    model = model.to(device)
    
    # Wrap model with DataParallel for multi-GPU training
    if use_multi_gpu:
        model = nn.DataParallel(model)
        print(f"Model wrapped with DataParallel for {num_gpus} GPUs")
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)

    initial_epoch = 0
    global_step = 0
    
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model weights from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
        # Handle loading weights for DataParallel models
        if use_multi_gpu and not list(state['model_state_dict'].keys())[0].startswith('module.'):
            # If current model is DataParallel but saved model wasn't, add 'module.' prefix
            model_state_dict = {'module.' + k: v for k, v in state['model_state_dict'].items()}
            model.load_state_dict(model_state_dict)
        elif not use_multi_gpu and list(state['model_state_dict'].keys())[0].startswith('module.'):
            # If current model is not DataParallel but saved model was, remove 'module.' prefix
            model_state_dict = {k.replace('module.', ''): v for k, v in state['model_state_dict'].items()}
            model.load_state_dict(model_state_dict)
        else:
            # Normal loading
            model.load_state_dict(state['model_state_dict'])
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)  # label smoothing makes the model less confident, which overall reduces overfitting and improves model accuracy
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        epoch_losses = []  # Track losses for this epoch
        
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            
            # Use the underlying model for custom methods when using DataParallel
            if use_multi_gpu:
                encoder_output = model.module.encode(encoder_input, encoder_mask)
                decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.module.project(decoder_output)
            else:
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
            
            label = batch['label'].to(device)

            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Track the loss
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            metrics_tracker.add_train_loss(loss_value, global_step)
            
            batch_iterator.set_postfix({"loss": loss_value})
            
            writer.add_scalar('Train Loss', loss_value, global_step)
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1

        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        
        # Save the underlying model state (without DataParallel wrapper)
        model_state_dict = model.module.state_dict() if use_multi_gpu else model.state_dict()
        
        # Run validation at the end of each epoch
        print(f'\n--- Running validation for epoch {epoch+1} ---')
        # Use the underlying model for validation (without DataParallel wrapper)
        eval_model = model.module if use_multi_gpu else model
        bleu_score = run_validation(eval_model, val_dataloader, tokenizer_src, tokenizer_tgt, 
                                  config['seq_len'], device, lambda msg: print(msg), 
                                  global_step, writer, num_examples=3)
        
        # Add metrics to tracker
        metrics_tracker.add_epoch_metrics(epoch + 1, bleu_score)
        
        # Calculate and print epoch statistics
        avg_epoch_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch+1} completed:')
        print(f'  Average Loss: {avg_epoch_loss:.4f}')
        print(f'  BLEU Score: {bleu_score:.4f}')
        print(f'  Batches processed: {len(epoch_losses)}')
        
        # Plot metrics every few epochs (to avoid too many plots)
        if (epoch + 1) % max(1, config['num_epochs'] // 5) == 0 or epoch == config['num_epochs'] - 1:
            print(f"Generating metrics plots...")
            try:
                metrics_tracker.plot_metrics(save_plots=True)
                if len(metrics_tracker.batch_losses) > 50:  # Only plot detailed loss if we have enough data
                    metrics_tracker.plot_detailed_loss(save_plots=True)
            except Exception as e:
                print(f"Warning: Could not generate plots: {e}")
        
        print()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)
        
    # After training is complete, run a final validation on test set
    print('\n' + '='*80)
    print('TRAINING COMPLETED - Running final validation on test set')
    print('='*80)
    
    # Load the best model (last saved model)
    best_model_filename = get_weights_file_path(config, f'{config["num_epochs"]-1:02d}')
    print(f"Loading best model from {best_model_filename}")
    state = torch.load(best_model_filename)
    
    # Handle loading weights for DataParallel models
    if use_multi_gpu:
        model.module.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state['model_state_dict'])
    
    # Run final validation on test set
    eval_model = model.module if use_multi_gpu else model
    final_bleu = run_validation(eval_model, test_dataloader, tokenizer_src, tokenizer_tgt, 
                              config['seq_len'], device, lambda msg: print(msg), 
                              global_step, writer, num_examples=5)
    
    print(f'\nFINAL TEST BLEU SCORE: {final_bleu:.4f}')
    
    # Generate final plots and save metrics
    print("\n" + "="*60)
    print("GENERATING FINAL TRAINING METRICS PLOTS AND REPORTS")
    print("="*60)
    
    try:
        # Save metrics data
        metrics_tracker.save_metrics_data()
        
        # Generate final plots
        metrics_tracker.plot_metrics(save_plots=True)
        metrics_tracker.plot_detailed_loss(save_plots=True)
        
        # Print training summary
        print("\n" + "-"*50)
        print("TRAINING SUMMARY")
        print("-"*50)
        if metrics_tracker.train_losses:
            print(f"Initial training loss: {metrics_tracker.train_losses[0]:.4f}")
            print(f"Final training loss: {metrics_tracker.train_losses[-1]:.4f}")
            print(f"Best training loss: {min(metrics_tracker.train_losses):.4f}")
        
        if metrics_tracker.validation_bleu_scores:
            print(f"Initial BLEU score: {metrics_tracker.validation_bleu_scores[0]:.4f}")
            print(f"Final BLEU score: {metrics_tracker.validation_bleu_scores[-1]:.4f}")
            print(f"Best BLEU score: {max(metrics_tracker.validation_bleu_scores):.4f}")
        
        print(f"Final test BLEU score: {final_bleu:.4f}")
        print(f"Total epochs: {config['num_epochs']}")
        print(f"Total batches processed: {len(metrics_tracker.batch_losses)}")
        print("-"*50)
        
        print("\n✅ Training completed successfully!")
        print("📊 Plots and metrics have been saved to the 'plots' directory")
        
    except Exception as e:
        print(f"⚠️  Error generating final plots/reports: {e}")
        print("Training completed but plots may not be available")
    
    print('='*80)
        

if __name__ == "__main__":
    # warnings.filterwarnings("ignore", category=UserWarning)  # Suppress specific user warnings
    config = get_config()
    train_model(config)
