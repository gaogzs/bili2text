from __future__ import annotations

import io
import importlib.util
import sys
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
import urllib.request

from b2t.i18n import dependency_sync_guidance
from b2t.transcribers.base import Transcriber


_FUNASR_REMOTE_FILES = {
    Path("model.py"): "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/model.py",
    Path("ctc.py"): "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/ctc.py",
    Path("tools/__init__.py"): None,
    Path("tools/utils.py"): "https://raw.githubusercontent.com/FunAudioLLM/Fun-ASR/main/tools/utils.py",
}


class FunASRTranscriber(Transcriber):
    name = "funasr"

    def __init__(
        self,
        *,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        language: str = "中文",
        use_itn: bool = True,
        hub: str = "hf",
        device: str | None = None,
    ) -> None:
        self.model_name = model
        self.language = language
        self.use_itn = use_itn
        self.hub = hub
        self.device = device
        self._model: Any | None = None
        self._runtime_kwargs: dict[str, Any] | None = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        prompt: str | None = None,
        progress=None,
    ) -> dict[str, Any]:
        model = self._ensure_model()
        if progress is not None:
            progress.running(
                "transcribing",
                message="transcribing",
                indeterminate=True,
                detail={"device": self.device or "unknown"},
            )

        try:
            result = self._run_inference(model, audio_path, prompt)
        except Exception as exc:
            if _is_cuda_oom(exc) and self.device and self.device.startswith("cuda"):
                # OOM on long audio is common; transparently retry once on CPU.
                if progress is not None:
                    progress.running(
                        "transcribing",
                        message="transcribing",
                        indeterminate=True,
                        detail={"device": "cpu", "fallback": "cuda_oom"},
                    )
                _clear_cuda_cache()
                self.device = "cpu"
                self._model = None
                self._runtime_kwargs = None
                model = self._ensure_model()
                try:
                    result = self._run_inference(model, audio_path, prompt)
                except Exception as retry_exc:
                    raise RuntimeError(f"Fun-ASR transcription failed after CPU retry: {retry_exc}") from retry_exc
            else:
                raise RuntimeError(f"Fun-ASR transcription failed: {exc}") from exc

        if isinstance(result, tuple) and len(result) == 2:
            result = result[0]
        if isinstance(result, list) and result:
            text_source = result[0]
            if isinstance(text_source, list) and text_source:
                text_source = text_source[0]
        else:
            text_source = result

        text = _extract_text(text_source).strip()
        return {
            "text": text,
            "segments": result,
            "language": self.language,
            "model": self.model_name,
            "device": self.device,
            "hub": self.hub,
        }

    def _run_inference(self, model: Any, audio_path: Path, prompt: str | None) -> Any:
        generate_kwargs: dict[str, Any] = {
            "language": self.language,
            "itn": self.use_itn,
        }
        hotwords = _prompt_to_hotwords(prompt)
        if hotwords:
            generate_kwargs["hotwords"] = hotwords

        if hasattr(model, "generate"):
            return model.generate(
                input=[str(audio_path)],
                cache={},
                batch_size=1,
                **generate_kwargs,
            )

        runtime_kwargs = self._runtime_kwargs or {}
        return model.inference(data_in=[str(audio_path)], **runtime_kwargs, **generate_kwargs)

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from funasr import AutoModel
            from funasr.download.download_model_from_hub import download_model
        except ImportError as exc:
            raise RuntimeError(build_funasr_import_error_message()) from exc

        if self.device is None:
            # Default to CPU for stability on long-form transcription.
            # Users can still force CUDA by setting `funasr.device` in config.
            self.device = "cpu"

        try:
            downloaded_kwargs = download_model(
                model=self.model_name,
                hub=self.hub,
                trust_remote_code=False,
            )
            downloaded_kwargs["device"] = self.device
            _ensure_funasr_remote_code()
            self._model, self._runtime_kwargs = _build_model_with_filtered_warnings(AutoModel, downloaded_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Fun-ASR model '{self.model_name}': {exc}") from exc
        return self._model


def build_funasr_import_error_message(*, funasr_available: bool | None = None) -> str:
    if funasr_available is None:
        funasr_available = importlib.util.find_spec("funasr") is not None

    if funasr_available:
        return (
            "Fun-ASR is installed, but the Python environment looks broken. "
            "Recreate `.venv` and sync the required extras again. "
            f"{dependency_sync_guidance('en-US')}"
        )

    return (
        "Fun-ASR support is not installed. "
        f"{dependency_sync_guidance('en-US')}"
    )


def _prompt_to_hotwords(prompt: str | None) -> list[str] | None:
    if not prompt:
        return None

    hotwords = [chunk.strip() for chunk in re.split(r"[\n,;，；、|]+", prompt) if chunk.strip()]
    return hotwords or None


def _extract_text(value: object) -> str:
    if isinstance(value, dict):
        text = value.get("text")
        if text is not None:
            return str(text)
        return "\n".join(_extract_text(item) for item in value.values()).strip()

    if isinstance(value, (list, tuple)):
        return "\n".join(_extract_text(item) for item in value if item is not None).strip()

    return str(value)


def _detect_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _ensure_funasr_remote_code() -> None:
    cache_root = Path.home() / ".cache" / "bili2text" / "funasr_remote_code"
    cache_root.mkdir(parents=True, exist_ok=True)

    for relative_path, url in _FUNASR_REMOTE_FILES.items():
        destination = cache_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)

        if url is None:
            if not destination.exists():
                destination.write_text("", encoding="utf-8")
            continue

        if not destination.exists():
            urllib.request.urlretrieve(url, destination)

    if str(cache_root) not in sys.path:
        sys.path.insert(0, str(cache_root))

    _import_module_from_file(cache_root / "ctc.py", "ctc")
    _import_module_from_file(cache_root / "tools" / "__init__.py", "tools")
    _import_module_from_file(cache_root / "tools" / "utils.py", "tools.utils")
    _import_module_from_file(cache_root / "model.py", "bili2text_funasr_model")


def _import_module_from_file(file_path: Path, module_name: str) -> None:
    if module_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Fun-ASR remote code from {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def _build_model_with_filtered_warnings(auto_model: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    captured = io.StringIO()
    with redirect_stdout(captured):
        model, runtime_kwargs = auto_model.build_model(**kwargs)

    lines = captured.getvalue().splitlines()
    noisy_prefixes = (
        "Warning, miss key in ckpt: ctc_decoder.",
        "Warning, miss key in ckpt: ctc.",
    )
    for line in lines:
        if line.startswith(noisy_prefixes):
            continue
        if line.strip():
            print(line)

    return model, runtime_kwargs


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message and "cuda" in message


def _clear_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()