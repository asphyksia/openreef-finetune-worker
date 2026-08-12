# Third-party notices

The OpenReef finetune worker is distributed under the MIT license. Its container
images install third-party Python and system packages under their own licenses.
`/licenses/python-distributions.json` records the installed Python distributions,
versions, declared licenses, and upstream project URLs for each concrete image.

Important training components include:

| Component | License | Usage in OpenReef |
| --- | --- | --- |
| Unsloth core (`unsloth/*`) | Apache-2.0 | NVIDIA single-GPU model loading and SFT |
| Unsloth Zoo | LGPL-3.0 | Runtime dependency of Unsloth core |
| Unsloth Studio and CLI directories | AGPL-3.0 | Not imported, copied, or modified by OpenReef application code |
| Axolotl | Apache-2.0 | AMD ROCm and NVIDIA multi-GPU SFT |
| Transformers / TRL / PEFT / Accelerate | Apache-2.0 | Training stack |
| PyTorch | BSD-style | Tensor runtime |
| bitsandbytes | MIT | Quantization and optimizer support where enabled |

The published `unsloth` Python distribution can include source directories under
different licenses. OpenReef calls the Apache-2.0 core API only, but redistributors
must still preserve all license files shipped by upstream packages. OpenReef does
not reuse Unsloth Studio or Unsloth CLI source code. Review upstream licenses when
changing, modifying, or redistributing those components.

Upstream sources:

- https://github.com/unslothai/unsloth
- https://github.com/unslothai/unsloth-zoo
- https://github.com/axolotl-ai-cloud/axolotl
- https://github.com/huggingface/transformers
- https://github.com/huggingface/trl
- https://github.com/pytorch/pytorch
