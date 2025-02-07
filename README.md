# LM Continual Pretraining Framework

**A Modular Toolkit for Efficient Language Model Pretraining and Adaptation**  
*Streamlining continual pretraining of foundation language models through scalable pipelines and reproducible configurations*

## 🌟 Key Features  
- **Resumable Training** - Checkpointing and state management for seamless continuation  
- **Distributed Optimization** - Integrated FSDP/DeepSpeed configurations

## Desirables!
- **Multi-format Support** - Unified interface for text corpora (JSONL, Parquet, TXT)
- **Curriculum Learning** - Dynamic data mixing strategies via `config/curricula`  
- **Health Monitoring** - Training telemetry and model diagnostics (`utils/monitoring.py`)  
- **Reproducibility** - Full experiment tracking with Hydra (`config/train.yaml`)  
