# Few-shot Fine-grained Image Classification with Interpretable Prompt Learning through Distribution Alignment (ICMI 2025)

## How to Install
This code is built on top of the awesome toolbox [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch) so you need to install the `dassl` environment first. Simply follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install `dassl` as well as PyTorch. After that, run `pip install -r requirements.txt` under `CoOp/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

## How to Run

### Generate Descriptions
Please refer to the `gpt4-v.py` to generate target few-shot descriptors.

### Few-Shot Learning

You can refer to the `food_coop_b16.sh` as an example.

`DATASET` takes as input a dataset name, like `food101`. The valid names are the files' names in `configs/datasets/`.

`CFG` means which config file to use, such as `rn50`, `rn101` or `vit_b32` (see `configs/trainers/CoOp/`).



## Citation
If you use this code in your research, please kindly cite the following papers

```bash
@inproceedings{guo2025few,
  title={Few-shot Fine-grained Image Classification with Interpretable Prompt Learning through Distribution Alignment},
  author={Guo, Dongliang and Zhao, Handong and Rossi, Ryan and Kim, Sungchul and Lipka, Nedim and Yu, Tong and Li, Sheng},
  booktitle={Proceedings of the 27th International Conference on Multimodal Interaction},
  pages={663--672},
  year={2025}
}
```
