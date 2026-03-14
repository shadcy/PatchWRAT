import pytest
import torch
import torch.nn as nn

# Assuming your classes are imported from a file named wrat.py
from wrat import LearnableDWT1D, FrequencySparseAttention, WRATBlock, WRATModel, WaveletTransformerLoss, LearnableTauWRATBlock

# ==========================================
# Fixtures (Shared Test Configurations)
# ==========================================
@pytest.fixture
def base_config():
    return {
        "batch_size": 2,
        "in_channels": 1,
        "d_model": 16,     # Kept small for fast testing
        "seq_len": 32,     # Must be even for stride=2 DWT
        "num_heads": 2,
    }

@pytest.fixture
def sample_input(base_config):
    torch.manual_seed(42) # Reproducibility
    return torch.randn(
        base_config["batch_size"], 
        base_config["in_channels"], 
        base_config["seq_len"]
    )

# ==========================================
# 1. Component Shape & Identity Tests
# ==========================================
def test_dwt_1d_shapes(base_config, sample_input):
    """Tests if DWT correctly halves the sequence and inverse restores it."""
    dwt = LearnableDWT1D(
        in_channels=base_config["in_channels"], 
        out_channels=base_config["d_model"]
    )
    
    LL, LH = dwt(sample_input)
    
    expected_len = base_config["seq_len"] // 2
    assert LL.shape == (base_config["batch_size"], base_config["d_model"], expected_len)
    assert LH.shape == (base_config["batch_size"], base_config["d_model"], expected_len)
    
    recon = dwt.inverse(LL, LH)
    assert recon.shape == sample_input.shape

def test_sparse_attention_shapes(base_config):
    """Tests attention mechanism output shapes."""
    seq_len_half = base_config["seq_len"] // 2
    attn = FrequencySparseAttention(
        d_model=base_config["d_model"], 
        num_heads=base_config["num_heads"]
    )
    
    # Input to attention is (B, L, D)
    qkv = torch.randn(base_config["batch_size"], seq_len_half, base_config["d_model"])
    out = attn(qkv, qkv, qkv)
    
    assert out.shape == qkv.shape

def test_wrat_block_shapes(base_config):
    """Tests if the core block maintains tensor dimensionality."""
    seq_len_half = base_config["seq_len"] // 2
    block = WRATBlock(d_model=base_config["d_model"], num_heads=base_config["num_heads"])
    
    # Input from DWT is (B, D, L)
    LL = torch.randn(base_config["batch_size"], base_config["d_model"], seq_len_half)
    LH = torch.randn(base_config["batch_size"], base_config["d_model"], seq_len_half)
    
    LL_out, LH_out = block(LL, LH)
    
    assert LL_out.shape == LL.shape
    assert LH_out.shape == LH.shape

# ==========================================
# 2. Integration & Gradient Flow Tests
# ==========================================
def test_wrat_model_forward(base_config, sample_input):
    """Tests full end-to-end forward pass for both fixed and learnable tau."""
    for tau_type in ['fixed', 'learnable']:
        model = WRATModel(
            in_channels=base_config["in_channels"],
            d_model=base_config["d_model"],
            num_heads=base_config["num_heads"],
            tau_type=tau_type
        )
        preds = model(sample_input)
        # Expected output shape should match input shape for sequence-to-sequence
        assert preds.shape == sample_input.shape

def test_wrat_model_zero_lh_flag(base_config, sample_input):
    """Verifies the zero_lh flag executes without crashing."""
    model = WRATModel(
        in_channels=base_config["in_channels"],
        d_model=base_config["d_model"],
        num_heads=base_config["num_heads"]
    )
    preds = model(sample_input, zero_lh=True)
    assert preds.shape == sample_input.shape

def test_wrat_backward_pass(base_config, sample_input):
    """
    CRITICAL TEST: Ensures gradients flow backward through all layers 
    without breaking or returning None.
    """
    model = WRATModel(
        in_channels=base_config["in_channels"],
        d_model=base_config["d_model"],
        num_heads=base_config["num_heads"],
        tau_type='learnable' # Test the most complex variant
    )
    loss_fn = WaveletTransformerLoss()
    
    # Forward pass
    preds = model(sample_input)
    targets = torch.randn_like(preds) # Mock targets
    
    # Get arbitrary recon for the custom loss requirement
    LL, LH = model.dwt(sample_input)
    recon = model.dwt.inverse(LL, LH)
    
    # Calculate loss
    total_loss, _, _, _ = loss_fn(preds, targets, sample_input, recon, model.dwt)
    
    # Backward pass
    total_loss.backward()
    
    # Check if critical components received gradients
    assert model.dwt.h.grad is not None, "DWT filter 'h' did not receive gradients."
    assert model.block._block.intra_LL_attn.q_proj.weight.grad is not None, "Attention layer missed gradients."
    assert model.block.raw_tau.grad is not None, "Learnable tau did not receive gradients."