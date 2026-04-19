from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from b2t.config import Settings
from b2t.i18n import DEFAULT_LANGUAGE, normalize_language

ALL_PROVIDERS = ("whisper", "sensevoice", "funasr", "volcengine")
ALL_FEATURES = ("web", "server", "window")


@dataclass(slots=True)
class SenseVoiceConfig:
    model_dir: str = ""
    language: str = "auto"
    use_itn: bool = True


@dataclass(slots=True)
class FunASRConfig:
    model: str = "FunAudioLLM/Fun-ASR-Nano-2512"
    language: str = "中文"
    use_itn: bool = True
    hub: str = "hf"
    device: str = ""


@dataclass(slots=True)
class VolcengineConfig:
    api_key: str = ""
    app_key: str = ""
    access_key: str = ""
    resource_id: str = "volc.bigasr.auc_turbo"
    model_name: str = "bigmodel"
    use_itn: bool = True


@dataclass(slots=True)
class AppConfig:
    language: str = DEFAULT_LANGUAGE
    enabled_providers: list[str] = field(default_factory=lambda: ["whisper"])
    enabled_features: list[str] = field(default_factory=lambda: ["window"])
    default_provider: str = "whisper"
    default_model: str = "small"
    sensevoice: SenseVoiceConfig = field(default_factory=SenseVoiceConfig)
    funasr: FunASRConfig = field(default_factory=FunASRConfig)
    volcengine: VolcengineConfig = field(default_factory=VolcengineConfig)

    @classmethod
    def load(cls, settings: Settings) -> "AppConfig":
        if not settings.config_path.exists():
            return cls()

        data = json.loads(settings.config_path.read_text(encoding="utf-8"))
        enabled = data.get("enabled_providers")
        if enabled is None:
            # backwards compat: old configs only had default_provider
            enabled = [data.get("default_provider", "whisper")]
        features = data.get("enabled_features", ["window"])
        return cls(
            language=normalize_language(data.get("language")),
            enabled_providers=enabled,
            enabled_features=features,
            default_provider=data.get("default_provider", "whisper"),
            default_model=data.get("default_model", "small"),
            sensevoice=SenseVoiceConfig(**data.get("sensevoice", {})),
            funasr=FunASRConfig(**data.get("funasr", {})),
            volcengine=VolcengineConfig(**data.get("volcengine", {})),
        )

    def save(self, settings: Settings) -> None:
        settings.ensure_directories()
        settings.config_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def default_model_for_provider(config: AppConfig, provider: str) -> str:
    selected_provider = provider.strip().lower()
    if selected_provider == "sensevoice":
        return config.sensevoice.model_dir or config.default_model or "small"
    if selected_provider == "funasr":
        return config.funasr.model or config.default_model or "small"
    if selected_provider == "volcengine":
        return config.volcengine.model_name or config.default_model or "small"
    return config.default_model or "small"
