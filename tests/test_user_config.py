from pathlib import Path

from b2t.config import Settings
from b2t.user_config import AppConfig, default_model_for_provider


def test_app_config_round_trip(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    config = AppConfig(
        default_provider="sensevoice",
        default_model="C:/models/sensevoice-small",
        language="en-US",
    )
    config.enabled_features = ["web", "window"]
    config.sensevoice.model_dir = "C:/models/sensevoice-small"
    config.volcengine.api_key = "secret"
    config.save(settings)

    loaded = AppConfig.load(settings)
    assert loaded.language == "en-US"
    assert loaded.enabled_features == ["web", "window"]
    assert loaded.default_provider == "sensevoice"
    assert loaded.default_model == "C:/models/sensevoice-small"
    assert loaded.sensevoice.model_dir == "C:/models/sensevoice-small"
    assert loaded.volcengine.api_key == "secret"


def test_default_model_for_provider_uses_provider_specific_values() -> None:
    config = AppConfig()
    config.sensevoice.model_dir = "C:/models/sensevoice-small"
    config.funasr.model = "FunAudioLLM/Fun-ASR-Nano-2512"
    config.volcengine.model_name = "bigmodel"

    assert default_model_for_provider(config, "sensevoice") == "C:/models/sensevoice-small"
    assert default_model_for_provider(config, "funasr") == "FunAudioLLM/Fun-ASR-Nano-2512"
    assert default_model_for_provider(config, "volcengine") == "bigmodel"
