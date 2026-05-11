from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_v1_prefix: str = "/api/v1"
    project_name: str = "Cephalometric Analysis API"

    class Config:
        env_file = ".env"

settings = Settings()