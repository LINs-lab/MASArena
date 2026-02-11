#!/usr/bin/env python3
"""
单测 whisper-1 是否可用。
用法:
  python scripts/test_whisper.py [可选: 音频文件路径]
不传路径时会在临时目录生成一段极短的静音 wav 用于测试接口。
"""
import os
import sys
import tempfile
import wave

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

def make_tiny_wav(path: str) -> None:
    """写一个极短的静音 wav，用于测试 API 是否连通。"""
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        # 0.1 秒静音
        w.writeframes(b"\x00\x00" * (8000 // 10))

def main() -> None:
    api_key = os.getenv("AUDIO_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("AUDIO_LLM_BASE_URL") or os.getenv("OPENAI_API_BASE")

    if not api_key:
        print("未设置 AUDIO_LLM_API_KEY 或 OPENAI_API_KEY")
        sys.exit(1)
    if not base_url:
        print("未设置 AUDIO_LLM_BASE_URL 或 OPENAI_API_BASE")
        sys.exit(1)

    print(f"API Base: {base_url}")
    print("调用 whisper-1 ...")

    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
        if not os.path.isfile(audio_path):
            print(f"文件不存在: {audio_path}")
            sys.exit(1)
        print(f"使用音频文件: {audio_path}")
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        try:
            make_tiny_wav(audio_path)
            print("使用临时静音 wav 测试")
        finally:
            pass  # 后面统一 delete

    try:
        import openai
        from openai import OpenAI
        
        # 检查文件大小
        file_size = os.path.getsize(audio_path)
        print(f"音频文件大小: {file_size} 字节")
        
        # 创建客户端，设置超时
        client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
            timeout=30.0,  # 30秒超时
        )
        
        # 读取文件内容
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        
        print(f"文件读取完成，准备上传...")
        
        # 使用文件对象而不是直接传递文件句柄，避免 multipart 解析问题
        from io import BytesIO
        audio_file = BytesIO(audio_data)
        audio_file.name = os.path.basename(audio_path)  # 设置文件名
        
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            timeout=30.0,  # 额外设置超时
        )
        
        text = transcription.text if hasattr(transcription, "text") else getattr(transcription, "text", str(transcription))
        print("whisper-1 调用成功")
        print("转录结果:", repr(text))
    except openai.APITimeoutError as e:
        print(f"❌ API 超时错误: {e}")
        print("提示: 代理服务器响应超时，可能是网络问题或代理配置问题")
        sys.exit(1)
    except openai.APIError as e:
        print(f"❌ API 错误: {e}")
        if hasattr(e, 'status_code'):
            print(f"状态码: {e.status_code}")
        if hasattr(e, 'response'):
            print(f"响应: {e.response}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ whisper-1 调用失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if len(sys.argv) < 2 and os.path.isfile(audio_path):
            os.unlink(audio_path)

if __name__ == "__main__":
    main()
