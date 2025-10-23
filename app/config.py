from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    service_mode: str = "txt2img"  # or "img2txt"
    hf_home: str = "./models"
    txt2img_model: str = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers"
    img2txt_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    img2txt_batch_size: int = 4
    img2txt_max_wait_ms: int = 5 * 1000


settings = Settings()
