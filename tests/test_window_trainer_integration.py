import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERL = ROOT / "external/revisiting_opd/verl"


def function(path, name):
    tree = ast.parse(path.read_text())
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def test_core_accepts_window_mode_and_offset_without_changing_default():
    fn = function(VERL / "trainer/ppo/core_algos.py", "compute_policy_loss")
    defaults = dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):], fn.args.defaults))
    assert "opd_window_mode" in defaults
    assert ast.literal_eval(defaults["opd_window_mode"]) == "fixed"
    assert ast.literal_eval(defaults["opd_window_offset"]) == 0


def test_actor_reads_offset_before_microbatch_shadowing():
    fn = function(VERL / "workers/actor/dp_actor.py", "update_policy")
    source = ast.unparse(fn)
    assert 'data.meta_info["opd_window_offset"]' in source or "data.meta_info['opd_window_offset']" in source
    assert "window_kwargs" in source
    assert "one optimizer step" in source


def test_driver_has_window_resume_contract_and_step_metadata():
    path = VERL / "trainer/ppo/ray_trainer_multitask.py"
    source = path.read_text()
    assert "window_state.json" in source
    assert "window_steps.jsonl" in source
    assert 'batch.meta_info["opd_window_offset"]' in source
    assert "_save_window_state(local_global_step_folder)" in source
    assert "_load_window_state(global_step_folder)" in source
    assert "window_credit_diagnostics" in source
    assert "window_ratio_diagnostics" in source


def test_window_diagnostics_are_labeled_as_loss_input_not_model_gradients():
    source = (VERL / "trainer/ppo/ray_trainer_multitask.py").read_text()
    assert '"gradient_diagnostic_space": "sampled_log_probs_not_model_parameters"' in source
