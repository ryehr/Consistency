This repository is the proof-of-concept code of the paper entitled "Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models".

## Repository Structure
```
📂 AddressingGlitch
├── 📂 Address_glitch_steganography_data/            # Experiment results on steganography
    ├── 📂 Basic                                     # Experiment results of the baseline Basic
    ├── 📂 MWIS                                      # Experiment results of the baseline MWIS
    ├── 📂 SyncPool                                  # Experiment results of the baseline SyncPool
    ├── 📂 Consistency                               # Experiment results of our method
    └──  📂 None                                     # Experiment results with error extractions
├── 📂 Address_glitch_data/                          # Experiment results on watermarking
    └──  📂 Attack/                                  # Experiment results on attacked watermarking
├── README.md                                   # Project overview
├── requirements.txt                            # Python dependencies
├── Steganography_AC.py                         # Code for steganography with arithmetic coding
├── Watermark_logits.py                         # Code for logit-based watermarking schemes
└── Watermark_sample.py                         # Code for sampling-based watermarking schemes
```

### Quick Usage 1
python Steganography_AC.py --disambiguation 'Basic' --top_k 128 --model_index 0 --sample_num 200
### Quick Usage 2
python Watermark_logits.py  --model_index 0 --glitch_period 2 --green_priority 2.0 --hash_window 1 --sample_num 200
### Quick Usage 3
python Steganography_sample.py --model_index 0 --glitch_period 2  --sample_num 500
