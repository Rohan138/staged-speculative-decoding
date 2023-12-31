<img src="./speculative-decoding.png" width="500px"></img>

## Staged Speculative Decoding
Implementation of [Staged Speculative Decoding](https://arxiv.org/abs/2308.04623)

## Instructions
```
git clone https://github.com/Rohan138/staged-speculative-decoding.git
pip install -r requirements.txt

# See main.py for additional arguments
python main.py --model gpt2-xl --draft-model distilgpt2 --dtype "float16" --num-samples 100
```

## Changelog
- 11/23/2023: Initial commit

## TODO
- [x] Choose and train draft and main models
- [x] Benchmark autoregressive decoding
- [x] Implement and benchmark naive speculative decoding
- [ ] Add tree-structured batches
- [ ] Add WanDB logging

## Acknowledgements

```bibtex
@inproceedings{
spector2023accelerating,
title={Accelerating {LLM} Inference with Staged Speculative Decoding},
author={Benjamin Frederick Spector and Christopher Re},
booktitle={Workshop on Efficient Systems for Foundation Models @ ICML2023},
year={2023},
url={https://openreview.net/forum?id=RKHF3VYjLK}
}
```

```bibtex
@inproceedings{Leviathan2022FastIF,
    title   = {Fast Inference from Transformers via Speculative Decoding},
    author  = {Yaniv Leviathan and Matan Kalman and Y. Matias},
    booktitle = {International Conference on Machine Learning},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:254096365}
}
```

```bibtex
@article{Chen2023AcceleratingLL,
    title     = {Accelerating Large Language Model Decoding with Speculative Sampling},
    author    = {Charlie Chen and Sebastian Borgeaud and Geoffrey Irving and Jean-Baptiste Lespiau and L. Sifre and John M. Jumper},
    journal   = {ArXiv},
    year      = {2023},
    volume    = {abs/2302.01318},
    url       = {https://api.semanticscholar.org/CorpusID:256503945}
}
```

```
@misc {gante2023assisted,
    author       = { {Joao Gante} },
    title        = { Assisted Generation: a new direction toward low-latency text generation },
    year         = 2023,
    url          = { https://huggingface.co/blog/assisted-generation },
    doi          = { 10.57967/hf/0638 },
    publisher    = { Hugging Face Blog }
}
```
