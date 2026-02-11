#!/usr/bin/env python3
"""
从Hugging Face下载模型到本地models文件夹的脚本
"""
import os
import argparse
from huggingface_hub import snapshot_download


def download_model(model_name, local_dir=None, token=None):
    """
    从Hugging Face下载模型

    Args:
        model_name: 模型名称，例如 'bert-base-chinese' 或 'THUDM/chatglm3-6b'
        local_dir: 本地保存路径，默认为 './models/{model_name}'
        token: Hugging Face访问令牌（如果需要下载私有模型）
    """
    # 如果没有指定本地路径，使用默认的models文件夹
    if local_dir is None:
        # 保持完整的模型路径结构，例如: ./models/THUDM/chatglm3-6b
        local_dir = os.path.join('models', model_name)

    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)

    print(f"开始下载模型: {model_name}")
    print(f"保存路径: {os.path.abspath(local_dir)}")

    try:
        # 下载模型
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,  # 支持断点续传
        )
        print(f"\n✓ 模型下载完成！")
        print(f"路径: {os.path.abspath(local_dir)}")

    except Exception as e:
        print(f"\n✗ 下载失败: {str(e)}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='从Hugging Face下载模型到本地',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 下载BERT中文模型（保存到 ./models/bert-base-chinese）
  python download_hf_model.py bert-base-chinese

  # 下载ChatGLM模型（保存到 ./models/THUDM/chatglm3-6b）
  python download_hf_model.py THUDM/chatglm3-6b

  # 指定自定义保存路径
  python download_hf_model.py bert-base-chinese --local-dir ./my_models/bert

  # 使用访问令牌下载私有模型
  python download_hf_model.py private-org/private-model --token YOUR_HF_TOKEN
        """
    )

    parser.add_argument(
        'model_name',
        type=str,
        help='Hugging Face模型名称，例如: bert-base-chinese 或 THUDM/chatglm3-6b'
    )

    parser.add_argument(
        '--local-dir',
        type=str,
        default=None,
        help='本地保存路径，默认为 ./models/{model_name}'
    )

    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face访问令牌（用于下载私有模型或提高下载速度）'
    )

    args = parser.parse_args()

    # 下载模型
    download_model(
        model_name=args.model_name,
        local_dir=args.local_dir,
        token=args.token
    )


if __name__ == '__main__':
    main()
