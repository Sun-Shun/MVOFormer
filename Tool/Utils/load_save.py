"""
load_save.py  --  checkpoint save/load with robust state-dict matching.

Key improvements over naive strict-loading:
  LS-1: Prints full summary: matched / missing / unexpected / shape-mismatch,
        instead of silently hiding missing keys.
  LS-2: Auto-handles 'module.' prefix mismatch between DDP-saved checkpoints
        and non-DDP models (and vice versa).
  LS-3: Truncates key lists to N entries to avoid MB-sized logs.
  LS-4: Reports match percentages and overall health.
  LS-5: load_model() kept for backward compat but delegates to the robust path.

Public API signatures are unchanged:
  get_checkpoint_state / save_checkpoint / load_checkpoint
"""
import os
import torch


# ── save (unchanged behaviour) ─────────────────────────────────────────────
def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(model=None, optimizer=None, epoch=None,
                         best_result=None, best_epoch=None,
                         test_loss=None, scaler_state=None):
    """
    scaler_state: GradScaler.state_dict(), non-None only for fp16 AMP.
    """
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        'epoch':           epoch,
        'model_state':     model_state,
        'optimizer_state': optim_state,
        'best_result':     best_result,
        'best_epoch':      best_epoch,
        'test_loss':       test_loss,
        'scaler_state':    scaler_state,
    }


def save_checkpoint(state, filename):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


# ── load (core fixes LS-1 ~ LS-4) ─────────────────────────────────────────
def _log(logger, level, msg):
    """Safe log helper (logger may be None or _NullLogger)."""
    if logger is None:
        return
    getattr(logger, level, logger.info)(msg)


def _sample(keys, n=10):
    keys = sorted(list(keys))
    if len(keys) <= n:
        return keys
    return keys[:n] + [f'... (+{len(keys) - n} more)']


def _reconcile_module_prefix(ckpt_state, model_state, logger=None):
    """
    Auto-handle 'module.' prefix mismatch.
    Returns (adjusted_ckpt_state, action_description).
    """
    ckpt_keys  = list(ckpt_state.keys())
    model_keys = list(model_state.keys())
    if len(ckpt_keys) == 0 or len(model_keys) == 0:
        return ckpt_state, "no-op (empty)"

    ckpt_has_module  = all(k.startswith('module.') for k in ckpt_keys)
    model_has_module = all(k.startswith('module.') for k in model_keys)

    if ckpt_has_module and not model_has_module:
        new_state = {k[len('module.'):]: v for k, v in ckpt_state.items()}
        _log(logger, 'info',
             "  prefix: stripped 'module.' from all ckpt keys (ckpt was DDP)")
        return new_state, "stripped 'module.' from ckpt"
    if model_has_module and not ckpt_has_module:
        new_state = {'module.' + k: v for k, v in ckpt_state.items()}
        _log(logger, 'info',
             "  prefix: added 'module.' to all ckpt keys (model is DDP)")
        return new_state, "added 'module.' to ckpt"
    return ckpt_state, "none (prefixes already match)"


def _smart_load_model_state(model, ckpt_state, logger=None, max_print=10):
    """
    Robustly load ckpt_state into model, matching by key name + shape.
    Keys with shape mismatch are skipped (model keeps its current values).
    Prints full summary: matched / unexpected / missing / shape_mismatch.
    """
    current_state = model.state_dict()

    # 1) prefix auto-adaptation
    ckpt_state, prefix_action = _reconcile_module_prefix(
        ckpt_state, current_state, logger=logger)

    # 2) match by name + validate shape
    matched = {}
    shape_mismatch = []   # (name, ckpt_shape, model_shape)
    for name, param in ckpt_state.items():
        if name in current_state:
            if current_state[name].shape == param.shape:
                matched[name] = param
            else:
                shape_mismatch.append(
                    (name, tuple(param.shape), tuple(current_state[name].shape)))

    unexpected_keys = set(ckpt_state.keys()) - set(current_state.keys())
    missing_keys    = set(current_state.keys()) - set(matched.keys())
    # Note: missing_keys includes shape_mismatch entries (not loaded)

    # 3) write back to model
    new_state = dict(current_state)
    new_state.update(matched)
    model.load_state_dict(new_state, strict=False)

    # 4) detailed summary
    n_ckpt  = len(ckpt_state)
    n_model = len(current_state)
    n_match = len(matched)
    pct_ckpt  = 100.0 * n_match / max(n_ckpt,  1)
    pct_model = 100.0 * n_match / max(n_model, 1)

    _log(logger, 'info', "=" * 74)
    _log(logger, 'info', "  Checkpoint load summary")
    _log(logger, 'info', f"  prefix handling:        {prefix_action}")
    _log(logger, 'info', f"  ckpt keys:              {n_ckpt}")
    _log(logger, 'info', f"  model keys:             {n_model}")
    _log(logger, 'info',
         f"  matched (loaded):       {n_match}  "
         f"({pct_ckpt:.1f}% of ckpt, {pct_model:.1f}% of model)")
    _log(logger, 'info', f"  shape-mismatch:         {len(shape_mismatch)}")
    _log(logger, 'info',
         f"  unexpected (ckpt-only): {len(unexpected_keys)}  "
         f"(ignored, did not load)")
    _log(logger, 'info',
         f"  missing    (model-only):{len(missing_keys)}  "
         f"<-- these keep random init")

    if shape_mismatch:
        _log(logger, 'warning', "  First shape mismatches (skipped):")
        for name, s_ckpt, s_model in shape_mismatch[:max_print]:
            _log(logger, 'warning',
                 f"    {name}: ckpt={s_ckpt}  model={s_model}")
        if len(shape_mismatch) > max_print:
            _log(logger, 'warning',
                 f"    ... (+{len(shape_mismatch) - max_print} more)")

    if unexpected_keys:
        _log(logger, 'warning', "  Sample unexpected keys (ignored):")
        for k in _sample(unexpected_keys, max_print):
            _log(logger, 'warning', f"    {k}")

    if missing_keys:
        _log(logger, 'warning',
             "  Sample missing keys (kept as random init):")
        for k in _sample(missing_keys, max_print):
            _log(logger, 'warning', f"    {k}")

    # health check
    if n_match == 0:
        _log(logger, 'error',
             "  !!! ZERO parameters matched -- checkpoint was NOT loaded. "
             "Likely a key-naming / module-wrapping mismatch. STOP and inspect.")
    elif n_match < n_ckpt * 0.5:
        _log(logger, 'warning',
             f"  !!! Only {pct_ckpt:.1f}% of ckpt keys matched. "
             "Confirm this is intentional.")
    _log(logger, 'info', "=" * 74)


def load_checkpoint(model, optimizer, filename, map_location, logger=None,
                    scaler=None):
    """
    Backward-compatible interface.
    - model:     can be None (optimizer-only restore)
    - optimizer: can be None (pretrain load, don't restore optimizer state)
    - scaler:    can be None. If non-None and ckpt has scaler_state, restore it.
    Returns: (epoch, best_result, best_epoch, test_loss)
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    _log(logger, 'info', f"==> Loading from checkpoint '{filename}'")
    # torch>=2.4 defaults weights_only=True; pass False for backward compat
    try:
        checkpoint = torch.load(filename, map_location=map_location,
                                weights_only=False)
    except TypeError:
        # older torch without the weights_only kwarg
        checkpoint = torch.load(filename, map_location=map_location)

    epoch       = checkpoint.get('epoch',       -1)
    best_result = checkpoint.get('best_result', 0.0)
    best_epoch  = checkpoint.get('best_epoch',  0)
    test_loss   = checkpoint.get('test_loss',   1.0)

    # ── model ──
    if model is not None and checkpoint.get('model_state') is not None:
        _smart_load_model_state(model, checkpoint['model_state'], logger=logger)
    elif model is not None:
        _log(logger, 'warning',
             "  checkpoint has no 'model_state' field, skipping model load.")

    # ── optimizer ──
    if optimizer is not None and checkpoint.get('optimizer_state') is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            _log(logger, 'info', "  optimizer state: loaded.")
        except Exception as e:
            _log(logger, 'warning',
                 f"  optimizer state: load FAILED ({type(e).__name__}: {e}). "
                 "Continuing with fresh optimizer state.")

    # ── GradScaler (fp16 resume) ──
    if scaler is not None and checkpoint.get('scaler_state') is not None:
        try:
            scaler.load_state_dict(checkpoint['scaler_state'])
            _log(logger, 'info', "  GradScaler state: loaded.")
        except Exception as e:
            _log(logger, 'warning',
                 f"  GradScaler state: load FAILED ({type(e).__name__}: {e})")

    _log(logger, 'info', "==> Done")
    return epoch, best_result, best_epoch, test_loss


def load_model(model, modelpath):
    """
    [deprecated] Kept for backward compat. Internally uses the robust loader:
      - auto 'module.' prefix handling
      - shape mismatch skip, strict=False
    """
    preTrainDict = torch.load(modelpath, map_location='cpu', weights_only=False)
    # support both raw state_dict and get_checkpoint_state dict
    if isinstance(preTrainDict, dict) and 'model_state' in preTrainDict:
        preTrainDict = preTrainDict['model_state']
    _smart_load_model_state(model, preTrainDict, logger=None)
    print('Model loaded via load_model (deprecated). See load_checkpoint() for detailed logging.')
    return model
