"""
配置 HuggingFace 镜像和 SSL 设置
解决国内网络环境下模型下载的 SSL 证书验证问题
"""

import os
import subprocess

def setup_huggingface_env():
    """设置 HuggingFace 环境变量"""
    
    HF_ENDPOINT = "https://hf-mirror.com"

    os.environ["HF_ENDPOINT"] = HF_ENDPOINT
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # 禁用 hf_transfer 加速
    
    print(f"已设置 HuggingFace 镜像: {HF_ENDPOINT}")
    
    try:
        import winreg
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment", 0, winreg.KEY_WRITE)
        winreg.SetValueEx(key, "HF_ENDPOINT", 0, winreg.REG_SZ, HF_ENDPOINT)
        winreg.CloseKey(key)
        print("已在系统环境变量中设置 HF_ENDPOINT，重启终端后生效")
    except Exception as e:
        print(f"无法写入系统环境变量: {e}")
        print("可在终端中手动运行以下命令设置环境变量:")
        print(f'  setx HF_ENDPOINT "{HF_ENDPOINT}"')
    
    return HF_ENDPOINT

def disable_ssl_verification():
    """禁用 SSL 验证（仅用于开发/测试环境，生产环境不推荐）"""
    import urllib.request
    import ssl
    
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    urllib.request.socket = None  # 这不会生效
    
    print("注意: SSL 验证已禁用")
    print("警告: 这仅适用于开发/测试环境，存在安全风险")
    
    return True

def fix_ssl_certificates():
    """修复 SSL 证书问题"""
    import certifi
    
    ca_bundle_path = certifi.where()
    print(f"certifi 证书路径: {ca_bundle_path}")
    
    os.environ["SSL_CERT_FILE"] = ca_bundle_path
    os.environ["REQUESTS_CA_BUNDLE"] = ca_bundle_path
    
    print("已设置 SSL 证书路径")
    return ca_bundle_path

if __name__ == "__main__":
    print("=" * 50)
    print("HuggingFace 环境配置")
    print("=" * 50)
    
    try:
        fix_ssl_certificates()
    except ImportError:
        print("certifi 未安装，跳过证书修复")
        print("运行: pip install certifi")
    
    setup_huggingface_env()
    
    print("\n配置完成！")
    print("请重启终端后运行:")
    print("  python main.py --project project_v1 --build-vector-index")
