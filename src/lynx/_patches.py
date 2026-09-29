"""Runtime monkey-patches that splice Lynx into vLLM without modifying
its source. Each patch is idempotent (re-running ``install_all()`` is a
no-op) and gated on ``VLLM_LYNX_ENABLED`` via :mod:`lynx._env`.

Patch summary:

1. ``FusedMoE.__init__`` (vLLM 0.20) or ``FusedMoEFactory`` (vLLM 0.30+) —
   inject the right Lynx routing variant.
2. ``GPUModelRunner.execute_model`` (both the V1 and the V2 runner) —
   per-batch prefill detection + ``LynxState.on_batch_start_worker``.
3. ``GPUModelRunner._dummy_run`` (both runners) — set ``is_prefill=False``
   so cudagraph capture exercises the Lynx kernel (else captured graphs
   bypass it).
4. ``Worker.__init__`` — load the policy JSON, inject onto ``hf_config``,
   create the per-worker ``LynxState`` singleton.
5. ``Worker.initialize_from_config`` — flip ``profile_complete=True``
   after kernel warmup.

The policy injection lives in the Worker patch (not a ModelConfig hook)
because plugin load order is more reliable there — by the time
``Worker.__init__`` runs, the worker has a fully-populated
``vllm_config`` with the correct ``hf_config`` available, and we are
guaranteed to be inside the worker process where ``LynxState`` needs to
be created anyway.

The patches are designed to fail loudly (with a logged warning) rather
than silently when the underlying vLLM API moves. They target vLLM 0.20.1
and 0.30.0.
"""

from __future__ import annotations

import json
import os
from typing import Any

from lynx._env import (
    config_file_override,
    is_enabled,
    metrics_enabled,
    profile_dir,
)

try:
    from vllm.logger import init_logger as _init_logger
    logger = _init_logger("lynx.patches")
except Exception:
    import logging
    logger = logging.getLogger(__name__)


_SENTINEL = "_lynx_patched"


def _already_patched(target: Any) -> bool:
    return getattr(target, _SENTINEL, False)


def install_all() -> None:
    """Apply every Lynx patch, in dependency order. Idempotent."""
    if not is_enabled():
        return
    _patch_fused_moe()
    _patch_gpu_worker()
    _patch_gpu_model_runner()
    logger.info("lynx: all patches installed")


def _resolve_lynx_policy(model_name: str) -> dict | None:
    """Load the active policy JSON. Tries ``VLLM_LYNX_CONFIG_FILE``
    first, then the bundled registry. Returns None if nothing matches."""
    from lynx.registry import lookup as registry_lookup

    path = config_file_override() or registry_lookup(model_name)
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. FusedMoE.__init__ — inject the right Lynx routing variant.
# ---------------------------------------------------------------------------


_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        logger.warning(msg)


def _select_lynx_routing(kwargs: dict[str, Any]) -> None:
    """Point a MoE layer's constructor kwargs at the matching Lynx router.

    Leaves ``kwargs`` untouched for layers Lynx cannot route correctly.
    """
    if kwargs.get("e_score_correction_bias") is not None:
        # Bias-corrected routing (DeepSeek-V3, GLM-4.5 and later) selects
        # experts on score + bias and scales the weights. The Lynx kernels
        # apply neither, so they would change which experts the model
        # picks. Keep these layers on vLLM's own router.
        _warn_once(
            "lynx: skipping MoE layers with bias-corrected routing "
            "(e_score_correction_bias); Lynx does not support it yet"
        )
        return
    # Import lazily so plugin load doesn't pay the triton-jit cost.
    if kwargs.get("use_grouped_topk", False):
        from lynx.routing import (
            custom_routing_function_grouped_topk as _crf,
        )
        # vLLM's router factory asserts that grouped top-k takes no custom
        # routing function, so the Lynx kernel does the whole selection.
        kwargs["use_grouped_topk"] = False
        kwargs["enable_eplb"] = False  # untested combo
    elif kwargs.get("scoring_func", "softmax") == "sigmoid":
        from lynx.routing import (
            custom_routing_function_sigmoid as _crf,
        )
        kwargs["enable_eplb"] = False
    else:
        from lynx.routing import custom_routing_function as _crf

    # Override any caller-supplied routing function (e.g. Llama4MoE).
    kwargs["custom_routing_function"] = _crf


def _patch_fused_moe() -> None:
    import functools

    from vllm.model_executor.layers.fused_moe import layer

    if not hasattr(layer, "FusedMoEFactory"):
        # vLLM <= 0.2x: MoE layers are FusedMoE instances.
        FusedMoE = layer.FusedMoE
        if _already_patched(FusedMoE.__init__):
            return
        _orig_init = FusedMoE.__init__

        @functools.wraps(_orig_init)
        def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            _select_lynx_routing(kwargs)
            return _orig_init(self, *args, **kwargs)

        _patched_init._lynx_patched = True  # type: ignore[attr-defined]
        FusedMoE.__init__ = _patched_init  # type: ignore[method-assign]
        logger.debug("lynx: patched FusedMoE.__init__")
        return

    # vLLM 0.30+ builds MoE layers through FusedMoEFactory. Models import it
    # from the package when they load, which happens after plugins install,
    # so rebinding both names reaches every model.
    import vllm.model_executor.layers.fused_moe as fused_moe_pkg

    if _already_patched(layer.FusedMoEFactory):
        return
    _orig_factory = layer.FusedMoEFactory

    @functools.wraps(_orig_factory)
    def _patched_factory(*args: Any, **kwargs: Any) -> Any:
        _select_lynx_routing(kwargs)
        return _orig_factory(*args, **kwargs)

    _patched_factory._lynx_patched = True  # type: ignore[attr-defined]
    layer.FusedMoEFactory = _patched_factory
    fused_moe_pkg.FusedMoEFactory = _patched_factory
    logger.debug("lynx: patched FusedMoEFactory")


# ---------------------------------------------------------------------------
# 2 + 3. GPUModelRunner: per-batch hook + dummy_run is_prefill flip.
# ---------------------------------------------------------------------------


def _runner_classes() -> list[type]:
    """GPU model runner classes in this vLLM. vLLM 0.30+ ships a second
    runner (``vllm.v1.worker.gpu.model_runner``) and uses it by default on
    NVIDIA, so patch both."""
    import importlib

    classes = []
    for mod in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker.gpu.model_runner",
    ):
        try:
            classes.append(importlib.import_module(mod).GPUModelRunner)
        except Exception as e:
            logger.debug("lynx: %s not patched: %s", mod, e)
    return classes


def _batch_counts(runner: Any, scheduler_output: Any) -> tuple[int, int, int, bool]:
    """Classify a batch as prefill or pure decode for LynxState."""
    num_new = len(getattr(scheduler_output, "scheduled_new_reqs", []) or [])
    num_chunked_prefill = 0
    num_pure_decode = 0
    is_prefill_batch = num_new > 0

    vllm_config = getattr(runner, "vllm_config", None)
    kvt = getattr(vllm_config, "kv_transfer_config", None)
    if is_prefill_batch and kvt is not None and getattr(kvt, "is_kv_consumer", False):
        # Disagg KV consumer: "new" requests are decode, KV cached upstream.
        is_prefill_batch = False
        num_pure_decode += num_new

    requests = getattr(runner, "requests", None)
    num_scheduled = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached is not None:
        for i, req_id in enumerate(getattr(cached, "req_ids", []) or []):
            req_state = requests.get(req_id) if requests is not None else None
            prompt = getattr(req_state, "prompt_token_ids", None)
            if prompt is not None:
                # V1 runner: compare progress against the prompt length.
                prefilling = cached.num_computed_tokens[i] < len(prompt)
            else:
                # V2 runner keeps no per-request prompt here. A decode step
                # schedules exactly one token per request.
                prefilling = num_scheduled.get(req_id, 1) > 1
            if prefilling:
                num_chunked_prefill += 1
                is_prefill_batch = True
            else:
                num_pure_decode += 1
    return num_new, num_pure_decode, num_chunked_prefill, is_prefill_batch


def _patch_gpu_model_runner() -> None:
    for runner_cls in _runner_classes():
        _patch_runner_class(runner_cls)


def _patch_runner_class(runner_cls: type) -> None:
    if _already_patched(runner_cls.execute_model):
        return

    _orig_execute = runner_cls.execute_model
    _orig_dummy = runner_cls._dummy_run

    def _patched_execute(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        # Dummy runs (profiling, cudagraph capture) carry an empty batch.
        # The _dummy_run patch has already set the state they need; a
        # per-batch update here would undo it.
        if not kwargs.get("dummy_run", False):
            try:
                from lynx.state import LynxState

                state = LynxState.get_instance()
                if state is not None:
                    state.on_batch_start_worker(*_batch_counts(self, scheduler_output))
            except Exception as e:  # never break the engine on telemetry hiccups
                logger.debug("lynx: per-batch hook skipped: %s", e)

        return _orig_execute(self, scheduler_output, *args, **kwargs)

    def _patched_dummy(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Force is_prefill=False, profile_complete=True so cudagraph
        # capture exercises the Lynx kernel. Without this, captured
        # graphs route through vanilla fused_topk and runtime replays
        # silently bypass Lynx — exactly the bug we hit in the in-tree
        # port.
        try:
            from lynx.state import LynxState

            state = LynxState.get_instance()
            if state is not None:
                state.is_prefill = False
                state.profile_complete = True
        except Exception:
            pass
        return _orig_dummy(self, *args, **kwargs)

    _patched_execute._lynx_patched = True  # type: ignore[attr-defined]
    _patched_dummy._lynx_patched = True  # type: ignore[attr-defined]
    runner_cls.execute_model = _patched_execute  # type: ignore[method-assign]
    runner_cls._dummy_run = _patched_dummy  # type: ignore[method-assign]
    logger.debug(
        "lynx: patched %s.%s.execute_model + _dummy_run",
        runner_cls.__module__,
        runner_cls.__name__,
    )


# ---------------------------------------------------------------------------
# 5 + 6. Worker.__init__ + Worker.initialize_from_config.
# ---------------------------------------------------------------------------


def _patch_gpu_worker() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    if _already_patched(Worker.__init__):
        return

    _orig_init = Worker.__init__
    _orig_initialize = Worker.initialize_from_config

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        _orig_init(self, *args, **kwargs)
        # Resolve and inject the Lynx policy into hf_config, then create
        # the per-worker LynxState. Done in the worker (not in a
        # ModelConfig hook) because plugin load order is more reliable
        # here: by this point self.vllm_config is fully populated.
        try:
            mc = self.vllm_config.model_config
            policy = _resolve_lynx_policy(mc.model)
            if policy is None:
                logger.warning(
                    "lynx: no policy registered for model %r; skipping. "
                    "Use lynx.register_model(...) or VLLM_LYNX_CONFIG_FILE.",
                    mc.model,
                )
                return
            for k, v in policy.items():
                setattr(mc.hf_config, k, v)

            from lynx.state import LynxState

            state = LynxState.create_instance(
                mc.hf_config,
                metrics_enabled=metrics_enabled(),
            )
            # Print to stderr (in addition to logger.info) so the
            # initialization is visible in vllm-serve logs even when
            # vllm's logging config silences third-party loggers.
            import sys as _sys
            print(
                f"lynx: state initialized on worker (pid={os.getpid()}, "
                f"policy={getattr(state, 'policy', None)}, "
                f"alpha={getattr(state, 'alpha', None)}, "
                f"beta={getattr(state, 'beta', None)})",
                file=_sys.stderr, flush=True,
            )
            if metrics_enabled():
                _create_metric_store(mc.hf_config)
        except Exception as e:
            logger.warning("lynx: LynxState creation failed: %s", e)

    def _patched_initialize(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = _orig_initialize(self, *args, **kwargs)
        # By this point kernel_warmup has run inside the original
        # initialize_from_config. Flip profile_complete so subsequent
        # forwards exercise the Lynx kernel.
        try:
            from lynx.state import LynxState

            state = LynxState.get_instance()
            if state is not None:
                state.mark_profiling_done()
                import sys as _sys
                print(
                    f"lynx: profiling complete on worker (pid={os.getpid()}); "
                    f"expert pruning is now active",
                    file=_sys.stderr, flush=True,
                )
            if metrics_enabled():
                _mark_metric_store_done()
        except Exception as e:
            logger.warning("lynx: profile-complete flip failed: %s", e)
        return result

    _patched_init._lynx_patched = True  # type: ignore[attr-defined]
    _patched_initialize._lynx_patched = True  # type: ignore[attr-defined]
    Worker.__init__ = _patched_init  # type: ignore[method-assign]
    Worker.initialize_from_config = _patched_initialize  # type: ignore[method-assign]
    logger.debug("lynx: patched Worker.__init__ + initialize_from_config")


# ---------------------------------------------------------------------------
# Optional metric setup (no-op when VLLM_LYNX_METRICS unset).
# ---------------------------------------------------------------------------


def _create_metric_store(hf_config: Any) -> None:
    """Lazy import — pulls in ddsketch only when metrics are enabled."""
    import atexit

    os.environ.setdefault("VLLM_LYNX_PROFILE_DIR", profile_dir())
    from lynx.metrics.metric_logging import MetricStore

    store = MetricStore.get_or_create_instance(hf_config)

    def _flush() -> None:
        try:
            store.plot_metrics()
        except Exception as e:  # pragma: no cover
            logger.warning("lynx metrics flush failed: %s", e)

    atexit.register(_flush)


def _mark_metric_store_done() -> None:
    from lynx.metrics.metric_logging import MetricStore

    store = MetricStore.get_instance()
    if store is not None:
        store.mark_profiling_done()
