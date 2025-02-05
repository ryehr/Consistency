This repository is the proof-of-concept code of the paper entitled "Addressing Glitch Tokens in Large Language Models for Steganography and Watermarking".

## Repository Structure
```
📂 AddressingGlitch
├── 📂 Address_glitch_steganography_data/                # Experiment results on steganography
    ├── 📂 Basic                                         # Experiment results of the baseline Basic
    ├── 📂 MWIS                                          # Experiment results of the baseline MWIS
    ├── 📂 SyncPool                                      # Experiment results of the baseline SyncPool
    ├── 📂 Consistency                                   # Experiment results of our method
    └──  📂 None                                          # Experiment results with error extractions
├── 📂 Address_glitch_data/                              # Experiment results on watermarking
    └──  📂 Attack/                                       # Experiment results on attacked watermarking
├── README.md                                           # Project overview
├── requirements.txt                                    # Python dependencies
├── Steganography_AC.py                                 # Code for steganography with arithmetic coding
├── Watermark_logits.py                                 # Code for logit-based watermarking schemes
└── Watermark_sample.py                                 # Code for sampling-based watermarking schemes
```
