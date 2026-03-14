# 1. Initialize the Lightning model
pl_model = WRATLightningModule(
    in_channels=1, 
    d_model=64, 
    tau_type='learnable',
    learning_rate=0.001
)

# 2. Setup the Trainer (automatically detects GPUs if available)
trainer = pl.Trainer(
    max_epochs=50,
    accelerator="auto", 
    devices="auto",
    logger=True # Enables TensorBoard logging by default
)

# 3. Train! (Assuming you have train_loader and val_loader)
# trainer.fit(pl_model, train_loader, val_loader)