"""Strict blueprint §9.3 configuration; no environment or backend discovery."""
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import math
import re
from typing import get_args, get_origin, get_type_hints

import yaml

from poc.contracts import Budget


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASES = {"exploratory", "confirmatory", "replication"}
COMMANDS = {"doctor", "validate", "prepare", "baselines", "diagnose", "fit", "compile", "export",
            "bench", "freeze", "evaluate", "analyze", "reproduce", "run", "report"}


class ConfigError(ValueError):
    """Invalid configuration, including unresolved consequential prerequisites."""


@dataclass(frozen=True)
class TeacherConfig:
    repository: str
    revision: str | None
    weights_sha256: str | None
    tokenizer_revision: str | None
    trust_remote_code: bool


@dataclass(frozen=True)
class NumericsConfig:
    scientific_backend: str
    scientific_dtype: str
    reference_backend: str
    reference_dtype: str
    deployment_backend: str
    loss_temperature: float
    head_position_chunk: int


@dataclass(frozen=True)
class ModelConfig:
    """Study settings; loaded-shape compatibility is checked by model adapters."""
    preserve_original_depth: bool
    share_attention: bool
    initial_target_bank_count: int
    ff_residual_rank: int
    residual_scale: float
    tied_embedding: bool


@dataclass(frozen=True)
class TrainingConfig:
    context_tokens: int
    microbatch_sequences: int
    target_valid_tokens_per_update: int
    bank_learning_rate: float
    residual_learning_rate: float
    adam_betas: tuple[float, float]
    nll_weight: float
    gradient_clip_norm: float
    final_target_presentations: int | None
    final_training_seeds: tuple[int, ...] | None


@dataclass(frozen=True)
class SearchConfig:
    seed: int
    natural_state_weight: float
    student_state_weight: float
    donor_state_weight: float
    teacher_query_budget: int | None
    total_search_seconds: float | None
    final_topology_policy: str | None


@dataclass(frozen=True)
class QuantizationConfig:
    initial_bits: int
    initial_group_size: int
    validated_formats: tuple[str, ...] | None
    calibration_manifest: Path | None
    preserve_canonical_sharing: bool


@dataclass(frozen=True)
class EvaluationConfig:
    primary_context_tokens: int
    primary_metric: str
    payload_budget_bytes: int | None
    protected_manifest: Path | None
    independent_cluster_count: int | None
    timing_pair_count: int | None
    analysis_freeze_sha256: str | None


@dataclass(frozen=True)
class BudgetConfig:
    """Unresolved pilot ceilings; units match Budget. No implicit defaults."""
    updates: int | None
    valid_target_presentations: int | None
    teacher_forward_calls: int | None
    suffix_backward_calls: int | None
    student_backward_calls: int | None
    search_proposals: int | None
    search_seconds: float | None
    wall_seconds: float | None
    disk_bytes: int | None
    memory_bytes: int | None

    def resolve(self) -> Budget:
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        missing = [name for name, value in values.items() if value is None]
        if missing:
            raise ConfigError(f"unresolved budgets: {', '.join(missing)}")
        return Budget(**values)


@dataclass(frozen=True)
class StudyConfig:
    schema_version: int
    study: str
    phase: str
    teacher: TeacherConfig
    numerics: NumericsConfig
    model: ModelConfig
    training: TrainingConfig
    search: SearchConfig
    quantization: QuantizationConfig
    evaluation: EvaluationConfig
    budgets: BudgetConfig


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"non-string or duplicate key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _typed(value, kind, name):
    args = get_args(kind)
    if type(None) in args:
        if value is None:
            return None
        return _typed(value, next(item for item in args if item is not type(None)), name)
    if hasattr(kind, "__dataclass_fields__"):
        if not isinstance(value, dict):
            raise ConfigError(f"{name} must be a mapping")
        expected = get_type_hints(kind)
        if set(value) != set(expected):
            raise ConfigError(f"{name}: unknown keys {sorted(set(value) - set(expected))}; missing keys {sorted(set(expected) - set(value))}")
        return kind(**{key: _typed(value[key], hint, f"{name}.{key}") for key, hint in expected.items()})
    if get_origin(kind) is tuple:
        if not isinstance(value, (list, tuple)) or not value:
            raise ConfigError(f"{name} must be a nonempty sequence")
        if args[-1] is not Ellipsis and len(value) != len(args):
            raise ConfigError(f"{name} has incorrect length")
        return tuple(_typed(item, args[0], name) for item in value)
    if kind is Path:
        if not isinstance(value, (str, Path)) or not str(value).strip():
            raise ConfigError(f"{name} must be a nonempty repository path")
        path = (PROJECT_ROOT / value).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ConfigError(f"{name} escapes repository root")
        return path
    if kind is float:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ConfigError(f"{name} must be a finite number")
        return float(value)
    if type(value) is not kind or (kind is str and not value.strip()):
        raise ConfigError(f"{name} must be {kind.__name__}")
    return value


def _range(name, value, *, minimum=0, maximum=None, inclusive=True):
    if value is None:
        return
    if (value < minimum if inclusive else value <= minimum) or (maximum is not None and value > maximum):
        raise ConfigError(f"{name} outside allowed range")


def _shape(config):
    if config.schema_version != 1 or config.phase not in PHASES:
        raise ConfigError("unsupported schema_version or phase")
    teacher, numerics, model, training, search, quant, evaluation = (
        config.teacher, config.numerics, config.model, config.training, config.search, config.quantization, config.evaluation)
    if teacher.repository not in {"HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M"} or teacher.trust_remote_code:
        raise ConfigError("teacher must be an approved local model with trust_remote_code=false")
    for name, value, length in [("teacher.revision", teacher.revision, 40), ("teacher.tokenizer_revision", teacher.tokenizer_revision, 40),
                                ("teacher.weights_sha256", teacher.weights_sha256, 64),
                                ("evaluation.analysis_freeze_sha256", evaluation.analysis_freeze_sha256, 64)]:
        if value is not None and not re.fullmatch(f"[0-9a-f]{{{length}}}", value):
            raise ConfigError(f"{name} must be a full immutable digest")
    if (numerics.scientific_backend not in {"torch_mps", "torch_cpu"} or numerics.scientific_dtype != "float32"
            or numerics.reference_backend != "cpu" or numerics.reference_dtype != "float64" or numerics.deployment_backend != "mlx"):
        raise ConfigError("unsupported scientific/reference/deployment numerical policy")
    if not model.preserve_original_depth or model.share_attention or not model.tied_embedding or not quant.preserve_canonical_sharing:
        raise ConfigError("primary track must preserve depth, attention, tying and canonical sharing")
    for name, actual, required in [
        ("numerics.loss_temperature", numerics.loss_temperature, 1.0),
        ("evaluation.primary_context_tokens", evaluation.primary_context_tokens, 512),
        ("model.residual_scale", model.residual_scale, 1.0),
    ]:
        if actual != required:
            raise ConfigError(f"{name} must be {required} for the primary study")
    positive = {"numerics.head_position_chunk": numerics.head_position_chunk,
                "model.initial_target_bank_count": model.initial_target_bank_count,
                "training.context_tokens": training.context_tokens,
                "training.microbatch_sequences": training.microbatch_sequences,
                "training.target_valid_tokens_per_update": training.target_valid_tokens_per_update,
                "training.bank_learning_rate": training.bank_learning_rate,
                "training.residual_learning_rate": training.residual_learning_rate,
                "training.gradient_clip_norm": training.gradient_clip_norm,
                "training.final_target_presentations": training.final_target_presentations,
                "evaluation.payload_budget_bytes": evaluation.payload_budget_bytes,
                "evaluation.independent_cluster_count": evaluation.independent_cluster_count,
                "evaluation.timing_pair_count": evaluation.timing_pair_count,
                "quantization.initial_group_size": quant.initial_group_size}
    for name, value in positive.items():
        _range(name, value, inclusive=False)
    for name, value in [("model.ff_residual_rank", model.ff_residual_rank),
                        ("training.nll_weight", training.nll_weight), ("search.seed", search.seed),
                        ("search.teacher_query_budget", search.teacher_query_budget), ("search.total_search_seconds", search.total_search_seconds)]:
        _range(name, value)
    for value in training.adam_betas:
        if not 0 <= value < 1:
            raise ConfigError("training.adam_betas must be in [0, 1)")
    for seed in training.final_training_seeds or ():
        _range("training.final_training_seeds", seed)
    weights = (search.natural_state_weight, search.student_state_weight, search.donor_state_weight)
    if any(not 0 <= weight <= 1 for weight in weights) or not math.isclose(sum(weights), 1, abs_tol=1e-12):
        raise ConfigError("search state weights must be in [0,1] and sum to one")
    if quant.initial_bits not in {2, 4, 8}:
        raise ConfigError("quantization.initial_bits must be 2, 4 or 8; support remains unqualified")
    if evaluation.primary_metric != "kl_teacher_to_model_nats_per_token":
        raise ConfigError("unsupported primary metric")
    for field in fields(config.budgets):
        _range(f"budgets.{field.name}", getattr(config.budgets, field.name))


def load_config(path: Path, *, phase: str) -> StudyConfig:
    try:
        raw = yaml.load((PROJECT_ROOT / path).read_text(), Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML: {exc}") from exc
    config = _typed(raw, StudyConfig, "config")
    _shape(config)
    if phase not in PHASES or config.phase != phase:
        raise ConfigError("requested phase must match configured phase")
    return config


def validate_config(config: StudyConfig, *, command: str) -> None:
    if command not in COMMANDS:
        raise ConfigError(f"unknown command: {command}")
    if not isinstance(config, StudyConfig):
        raise ConfigError("config must be a StudyConfig")
    config = _typed(asdict(config), StudyConfig, "config")
    _shape(config)
    if command in {"compile", "fit", "diagnose", "baselines", "bench", "freeze", "run", "evaluate"}:
        config.budgets.resolve()
        for field in fields(config.teacher):
            if getattr(config.teacher, field.name) is None:
                raise ConfigError(f"unresolved teacher.{field.name}")
    requirements = {
        "compile": [("search", "teacher_query_budget"), ("search", "total_search_seconds"),
                    ("search", "final_topology_policy"), ("evaluation", "payload_budget_bytes"),
                    ("training", "final_target_presentations"), ("training", "final_training_seeds")],
        "fit": [("training", "final_target_presentations"), ("training", "final_training_seeds")],
        "baselines": [("quantization", "calibration_manifest")],
    }
    if command in {"freeze", "run"} or (command == "evaluate") or config.phase in {"confirmatory", "replication"} and command not in {"doctor", "prepare", "validate", "report"}:
        requirements[command] = [(section.name, field.name) for section in fields(config)
                                 if hasattr(getattr(config, section.name), "__dataclass_fields__")
                                 for field in fields(getattr(config, section.name))
                                 if not (command == "freeze" and field.name == "analysis_freeze_sha256")]
    for section, name in requirements.get(command, []):
        if getattr(getattr(config, section), name) is None:
            raise ConfigError(f"unresolved {section}.{name} for {command}")
