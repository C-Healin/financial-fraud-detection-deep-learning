# GitHub 上传步骤

## 方法一：网页上传

1. 登录 GitHub，点击 **New repository**。
2. Repository name 建议填写：`financial-fraud-detection-deep-learning`。
3. Description 建议填写：
   `End-to-end financial fraud detection with XGBoost, PyTorch, GraphSAGE, FastAPI, Streamlit, Docker and NVIDIA Triton.`
4. 选择 Public。
5. 不要勾选自动生成 README、License 或 `.gitignore`，因为项目中已经包含。
6. 创建仓库后，选择 **uploading an existing file**，上传解压后的全部项目文件。

## 方法二：Git 命令行

在项目文件夹中执行：

```bash
git init
git add .
git commit -m "Build end-to-end financial fraud detection system"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/financial-fraud-detection-deep-learning.git
git push -u origin main
```

## 上传前建议修改

- 在 `README.md` 顶部加入你的 GitHub 用户名或 LinkedIn。
- 根据需要在 `pyproject.toml` 和 `LICENSE` 中确认英文姓名。
- 真实数据必须放在 `data/raw/`，不要上传到公开仓库。
- 当前 `data/sample/` 与 `artifacts/` 均为合成数据和演示模型，可以公开。

## GitHub 仓库建议设置

Topics：

```text
fraud-detection fintech machine-learning deep-learning pytorch xgboost graph-neural-networks graphsage fastapi streamlit docker nvidia-triton
```

在仓库页面右侧 About 区域勾选：

- Use your GitHub Pages website（仅在后续部署页面时）
- Releases
- Packages（后续发布 Docker 镜像时）
