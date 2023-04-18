# tdnn-on-directml

![GitHub last commit](https://img.shields.io/github/last-commit/konas122/tdnn-on-directml)
![GitHub license](https://img.shields.io/github/license/konas122/tdnn-on-directml?style=flat-square)

## Note

该仓库是在本人的另一个仓库 [konas122](https://github.com/konas122)/**[Voiceprint-recognition](https://github.com/konas122/Voiceprint-recognition)** 基础上构建的。

该仓库也是用于声纹识别，只是本仓库不是用来训练或与训练用的，而是用于 `tdnn` 模型的评估和实际应用。这是因为本仓库模型的运行是基于 [DirectML](https://github.com/microsoft/DirectML)。然而，如今的 **DirectML** 仍不支持模型的的反向传播 (以后也许会)，因此只能用于模型的评估或实际应用。



## requirement

```
python==3.8
numpy==1.23.5
librosa==0.9.2
scikit-learn==1.2.2
matplotlib==3.6.3
torch==1.13.1
torchaudio==0.13.1
torch-directml==0.1.13.1.dev230301
```



## Usage

先下载本人训练好的模型[releases](https://github.com/konas122/tdnn-on-directml/releases)，并将该模型放在 `run_this.py` 的同一目录下。

若要将模型直接运用，则运行 `run_this.py`。

若要对模型进行评估，则运行 `amd/amd_eval.py`。

​	

未完待续
