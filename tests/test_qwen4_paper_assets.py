"""All numeric fixtures are synthetic, never paper evidence."""

import copy
import importlib
import json

import numpy as np
import pytest


def test_module_available():
    assert importlib.util.find_spec("scripts.build_qwen4_paper_assets") is not None


@pytest.fixture
def m():
    return importlib.import_module("scripts.build_qwen4_paper_assets")


def source(path):
    return {"path": str(path), "sha256": "a" * 64}


@pytest.fixture
def bundle(m):
    data = {"schema_version": 1, "data_sha256": {task: "b" * 64 for task in m.TASKS}, "arms": {}}
    for variant in m.VARIANTS:
        arm = {
            "training_root": str(m.TRAIN_ROOTS[variant]),
            "evaluation_root": str(m.EVAL_ROOTS[variant]),
            "checkpoints": {},
        }
        for step in m.STEPS:
            model = str(m.EVAL_ROOTS[variant] / f"merged/{variant}_step{step}")
            folder = m.EVAL_ROOTS[variant] / f"evaluations/{variant}_step{step}"
            result = {
                "passed": True,
                "completion_protocol_verified": True,
                "model": model,
                "grader_sha256": m.GRADER_SHA,
                "rollout_archive": {"passed": True, "num_examples": 643, "num_rollouts": 5144},
                "sha256": {str(m.DATA / f"{task}.jsonl"): "b" * 64 for task in m.TASKS},
                "per_task": {
                    task: {
                        "num_examples": count,
                        "num_rollouts": count * 8,
                        "avg_at_8": 0.5,
                        "pass_at_8": 1.0,
                        "format_error_rollouts": count,
                        "format_error_rate": 0.125,
                        "engine_length_stop_rollouts": 0,
                        "engine_truncation_ratio": 0.0,
                    }
                    for task, count in m.TASKS.items()
                },
            }
            arm["checkpoints"][str(step)] = {
                "source": source(folder / "acceptance.json"),
                "result": result,
                "eval_card": {
                    **copy.deepcopy(m.EVAL_CONTRACT),
                    "variant": variant,
                    "checkpoint_step": step,
                    "model": model,
                    "source": source(folder / "eval_card.json"),
                },
            }
        data["arms"][variant] = arm
    return data


def add_questions(m, bundle, variant, step=200):
    cp = bundle["arms"][variant]["checkpoints"][str(step)]
    cp["questions"] = {
        "sources": [
            source(
                m.EVAL_ROOTS[variant]
                / f"evaluations/{variant}_step{step}/outputs/{task}_graded.jsonl"
            )
            for task in m.TASKS
        ],
        "rows": [
            {
                "task": task,
                "id": str(i),
                "correct_count": 4,
                "missing_box_count": 1,
                "length_stop_count": 0,
            }
            for task, count in m.TASKS.items()
            for i in range(count)
        ],
    }
    cp["result"]["sha256"].update(
        {item["path"]: item["sha256"] for item in cp["questions"]["sources"]}
    )
    return cp["questions"]["rows"]


def test_question_sources_must_be_the_accepted_graded_files(m, bundle):
    add_questions(m, bundle, "block3_mean")
    cp = bundle["arms"]["block3_mean"]["checkpoints"]["200"]
    cp["questions"]["sources"][0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="question source"):
        m.analyze(bundle)


def test_schema_explains_entropy_mask_and_credit_formula(m):
    definitions = m.schema()["definitions"]
    assert "full-vocabulary" in definitions["entropy"].lower()
    assert "surprisal" in definitions["entropy"]
    assert "151643" in definitions["response_mask"]
    assert "pre-update" in definitions["entropy_timing"]
    assert "not missing" in definitions["teacher_entropy_limit"]
    assert "epsilon=1e-4" in definitions["sign_flip_formula"]
    assert "abs" in definitions["leakage_formula"]


def test_duplicate_json_keys_fail_closed(m, bundle, tmp_path):
    raw = json.dumps(bundle)
    raw = raw.replace('"passed": true', '"passed": false, "passed": true', 1)
    path = tmp_path / "input.json"
    path.write_text(raw)
    with pytest.raises(ValueError, match="duplicate JSON"):
        m.build(path, tmp_path / "out", tmp_path / "figs", make_figures=False)
    assert not (tmp_path / "out").exists()


def test_full_bundle_preserves_four_tasks_and_checkpoint_order(m, bundle):
    report = m.analyze(bundle)
    assert report["complete"] is True and report["missing_checkpoints"] == []
    assert len(report["rows"]) == 32
    assert [r["step"] for r in report["rows"][:4]] == list(m.STEPS)
    assert {r["task"] for r in report["rows"]} == set(m.TASKS)
    assert report["bootstrap"] == []
    assert not any("macro" in key for key in report)


def test_original_block3_eval_card_without_variant_is_accepted(m, bundle):
    for checkpoint in bundle["arms"]["block3_mean"]["checkpoints"].values():
        del checkpoint["eval_card"]["variant"]
    assert m.analyze(bundle)["complete"]
    del bundle["arms"]["token_opd"]["checkpoints"]["200"]["eval_card"]["variant"]
    with pytest.raises(ValueError):
        m.analyze(bundle)


def test_nonfinite_unplotted_input_is_rejected_before_writes(m, bundle, tmp_path):
    bundle["arms"]["block3_mean"]["checkpoints"]["200"]["result"]["per_task"]["math500"][
        "mean_generated_tokens"
    ] = float("nan")
    path = tmp_path / "input.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="finite"):
        m.build(path, tmp_path / "out", tmp_path / "figs", make_figures=False)
    assert not (tmp_path / "out").exists()


def test_missing_checkpoint_fails_closed_or_is_explicitly_partial(m, bundle):
    del bundle["arms"]["token_opd"]["checkpoints"]["150"]
    with pytest.raises(ValueError, match="missing checkpoint"):
        m.analyze(bundle)
    report = m.analyze(bundle, allow_partial=True)
    assert report["complete"] is False
    assert report["missing_checkpoints"] == ["token_opd:150"]
    missing = [r for r in report["rows"] if r["variant"] == "token_opd" and r["step"] == 150]
    assert len(missing) == 4
    assert all(r["avg_at_8"] is None and r["status"] == "missing" for r in missing)


def test_absent_token_arm_is_not_fabricated(m, bundle):
    del bundle["arms"]["token_opd"]
    report = m.analyze(bundle, allow_partial=True)
    assert len(report["missing_checkpoints"]) == 4
    assert all(r["avg_at_8"] is None for r in report["rows"] if r["variant"] == "token_opd")


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("completion_protocol_verified", False),
        ("grader_sha256", "f" * 64),
        ("model", "/wrong/model"),
    ],
)
def test_unaccepted_or_wrong_result_is_never_hidden_by_partial(m, bundle, field, value):
    bundle["arms"]["token_opd"]["checkpoints"]["200"]["result"][field] = value
    with pytest.raises(ValueError):
        m.analyze(bundle, allow_partial=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_source_commit", "wrong"),
        ("prompt_protocol", "chat"),
        ("n", 1),
        ("temperature", 0),
        ("variant", "block3_mean"),
        ("enable_thinking", True),
        ("retain_rollouts", False),
    ],
)
def test_eval_contract_is_frozen(m, bundle, field, value):
    bundle["arms"]["token_opd"]["checkpoints"]["200"]["eval_card"][field] = value
    with pytest.raises(ValueError):
        m.analyze(bundle)


def test_dataset_identity_and_provenance_are_required(m, bundle):
    cp = bundle["arms"]["token_opd"]["checkpoints"]["200"]
    cp["result"]["sha256"][str(m.DATA / "aime24.jsonl")] = "c" * 64
    with pytest.raises(ValueError, match="data"):
        m.analyze(bundle)
    cp["result"]["sha256"][str(m.DATA / "aime24.jsonl")] = "b" * 64
    cp["source"]["sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="sha256"):
        m.analyze(bundle)


@pytest.mark.parametrize(
    "field,value",
    [
        ("avg_at_8", float("nan")),
        ("pass_at_8", -0.1),
        ("num_examples", 499),
        ("format_error_rate", 0.9),
        ("engine_length_stop_rollouts", 1),
    ],
)
def test_invalid_summary_or_denominator_is_rejected(m, bundle, field, value):
    bundle["arms"]["block3_mean"]["checkpoints"]["200"]["result"]["per_task"]["math500"][field] = (
        value
    )
    with pytest.raises(ValueError):
        m.analyze(bundle)


def test_bootstrap_pairs_question_ids_not_row_order(m, bundle):
    left = add_questions(m, bundle, "block3_mean")
    right = add_questions(m, bundle, "token_opd")
    for rows in (left, right):
        for row in rows:
            row["correct_count"] = 2 if int(row["id"]) % 2 else 6
    # Restore the odd-size AMC mean after deliberately making heterogeneous questions.
    for variant in m.VARIANTS:
        cp = bundle["arms"][variant]["checkpoints"]["200"]
        cp["result"]["per_task"]["amc23"]["avg_at_8"] = (42 * 6 + 41 * 2) / (83 * 8)
    right.reverse()
    report = m.analyze(bundle, bootstrap_replicates=64, bootstrap_seed=17)
    assert len(report["bootstrap"]) == 4
    for row in report["bootstrap"]:
        assert row["num_questions"] == m.TASKS[row["task"]]
        assert row["avg_at_8"]["lower_95_pp"] == row["avg_at_8"]["upper_95_pp"] == 0
        assert row["pass_at_8"]["delta_pp"] == 0
    assert "training-run" in report["uncertainty_scope"]
    assert report == m.analyze(bundle, bootstrap_replicates=64, bootstrap_seed=17)


def test_bootstrap_is_block_minus_token_and_task_local(m):
    left = [{"id": str(i), "correct_count": 8} for i in range(3)]
    right = [{"id": str(i), "correct_count": 0} for i in reversed(range(3))]
    result = m.paired_bootstrap(left, right, replicates=32, seed=1)
    assert result["avg_at_8"] == {"delta_pp": 100.0, "lower_95_pp": 100.0, "upper_95_pp": 100.0}
    assert result["pass_at_8"] == result["avg_at_8"]


def test_question_duplicates_mismatches_and_false_summaries_fail(m, bundle):
    left = add_questions(m, bundle, "block3_mean")
    right = add_questions(m, bundle, "token_opd")
    right[0]["id"] = "unmatched"
    with pytest.raises(ValueError, match="identit"):
        m.analyze(bundle)
    right[0]["id"] = right[1]["id"]
    with pytest.raises(ValueError, match="duplicate"):
        m.analyze(bundle)
    right[0]["id"] = left[0]["id"]
    right[0]["correct_count"] = 5
    with pytest.raises(ValueError, match="summary"):
        m.analyze(bundle)


def test_extra_checkpoint_and_raw_payload_are_rejected(m, bundle):
    cp = bundle["arms"]["token_opd"]["checkpoints"]
    cp["1"] = copy.deepcopy(cp["200"])
    with pytest.raises(ValueError):
        m.analyze(bundle, allow_partial=True)
    del cp["1"]
    cp["200"]["response_token_ids"] = [1, 2]
    with pytest.raises(ValueError, match="raw"):
        m.analyze(bundle)


def make_npz(path, *, bad=False):
    counts = np.zeros(16384, dtype=int)
    counts[:4] = [8, 2, 1, 1]
    sums = counts.astype(float) * np.arange(16384)
    if bad:
        counts[0] = -1
    np.savez(
        path,
        step=1,
        topk=16,
        position_stride=1,
        position_bin=128,
        gradient_diagnostic_space="sampled_log_probs_not_model_parameters",
        block_ratio_definition="post_update_policy_vs_behavior_policy",
        sign_epsilon=0.0001,
        prompt_batch_sha256="c" * 64,
        student_entropy__sum=sums,
        student_entropy__squared_sum=sums * np.arange(16384),
        student_entropy__valid_count=counts,
    )


def test_npz_compaction_uses_counts_and_preserves_coverage_not_average_of_means(m, tmp_path):
    path = tmp_path / "step_00001.npz"
    make_npz(path)
    item = m.compact_npz(path, bin_size=2)
    stats = item["metrics"]["student_entropy"]
    assert stats["valid_count"][:2] == [10, 2]
    assert stats["max_position_count"][:2] == [8, 1]
    values = m.position_means(item, "student_entropy", min_count=8)
    assert values[0] == pytest.approx(0.2)
    assert np.isnan(values[1:]).all()
    assert item["metadata"]["gradient_diagnostic_space"] == "sampled_log_probs_not_model_parameters"
    assert item["source"]["sha256"] == m.sha256(path)


def test_invalid_position_counts_rejected(m, tmp_path):
    path = tmp_path / "bad.npz"
    make_npz(path, bad=True)
    with pytest.raises(ValueError):
        m.compact_npz(path)


def test_scalar_records_reject_duplicates_and_nonfinite(m, bundle):
    arm = bundle["arms"]["block3_mean"]
    row = {
        "step": 1,
        "diagnostics/student_entropy": 0.2,
        "actor/grad_norm": 3.0,
        "response_length/clip_ratio": 0.125,
    }
    arm["scalars"] = {
        "source": source(m.TRAIN_ROOTS["block3_mean"] / "diagnostics/scalars.jsonl"),
        "records": [row, dict(row)],
    }
    with pytest.raises(ValueError, match="duplicate"):
        m.analyze(bundle)
    arm["scalars"]["records"] = [{**row, "actor/grad_norm": float("inf")}]
    with pytest.raises(ValueError, match="finite"):
        m.analyze(bundle)


def test_build_bilingual_tables_deterministic_and_refuses_overwrite(m, bundle, tmp_path):
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(bundle))
    output, figures = tmp_path / "generated", tmp_path / "figures"
    report = m.build(input_path, output, figures, make_figures=False)
    assert report["complete"]
    assert (output / "qwen4_results.csv").read_text().count("\n") == 33
    en = (output / "qwen4_results_en.tex").read_text()
    zh = (output / "qwen4_results_zh.tex").read_text()
    assert "Avg@8" in en and "MATH500" in en
    assert "\u7f3a\u5931" in zh
    assert "Pass@8 & No box & Trunc." in en
    assert "Pass@8 & \u7f3abox & \u622a\u65ad" in zh
    assert "Missing-box marker" not in en
    for table in (en, zh):
        assert r"\texttt{\textbackslash boxed}" in table
    provenance = json.loads((output / "qwen4_provenance.json").read_text())
    assert provenance["input_sha256"] == m.sha256(input_path)
    assert "qwen4_results.csv" in provenance["output_sha256"]
    with pytest.raises(FileExistsError):
        m.build(input_path, output, figures, make_figures=False)
    out2 = tmp_path / "repeat"
    m.build(input_path, out2, tmp_path / "figures2", make_figures=False)
    assert (output / "qwen4_results.csv").read_bytes() == (out2 / "qwen4_results.csv").read_bytes()


def test_invalid_input_writes_nothing(m, bundle, tmp_path):
    del bundle["arms"]["token_opd"]
    path = tmp_path / "input.json"
    path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError):
        m.build(path, tmp_path / "out", tmp_path / "figs")
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "figs").exists()


def test_symlink_output_rejected(m, bundle, tmp_path):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(bundle))
    old = tmp_path / "old"
    old.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(old, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        m.build(path, link / "new", tmp_path / "figs", make_figures=False)
    assert list(old.iterdir()) == []


def test_partial_plots_and_optional_diagnostics_render(m, bundle, tmp_path):
    import matplotlib.image as mpimg

    del bundle["arms"]["token_opd"]["checkpoints"]["50"]
    path = tmp_path / "step_00001.npz"
    make_npz(path)
    for variant in m.VARIANTS:
        arm = bundle["arms"][variant]
        arm["positions"] = [m.compact_npz(path, bin_size=128)]
        arm["positions"][0]["source"]["path"] = str(
            m.TRAIN_ROOTS[variant] / "diagnostics/step_00001.npz"
        )
        arm["scalars"] = {
            "source": source(m.TRAIN_ROOTS[variant] / "diagnostics/scalars.jsonl"),
            "records": [
                {
                    "step": step,
                    "diagnostics/student_entropy": 0.2 + step / 1000,
                    "diagnostics/teacher_entropy": 0.1,
                    "actor/grad_norm": 2.0,
                    "response_length/clip_ratio": 0.125,
                }
                for step in (1, 2, 4)
            ],
        }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(bundle))
    figures = tmp_path / "figures"
    figures.mkdir()
    concepts = [figures / f"{name}.png" for name in ("study_overview", "credit_assignment_concept")]
    for path in concepts:
        path.write_bytes(b"existing concept figure")
    m.build(input_path, tmp_path / "out", tmp_path / "figures", allow_partial=True)
    for stem in m.figure_stems(m.analyze(bundle, allow_partial=True)):
        image = mpimg.imread(tmp_path / "figures" / f"{stem}.png")
        assert np.std(image[..., :3]) > 0.02
        assert (tmp_path / "figures" / f"{stem}.pdf").stat().st_size > 1000
    assert "PARTIAL" in (tmp_path / "out/qwen4_results_en.tex").read_text()
    assert "NA" in (tmp_path / "out/qwen4_results_en.tex").read_text()
    assert all(path.read_bytes() == b"existing concept figure" for path in concepts)
    assert all(path in concepts or path.name.startswith("qwen4") for path in figures.iterdir())


def test_publication_figures_preserve_data_dimensions_and_readable_fonts(
    m, bundle, tmp_path, monkeypatch
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    del bundle["arms"]["token_opd"]["checkpoints"]["50"]
    path = tmp_path / "step_00001.npz"
    make_npz(path)
    for variant in m.VARIANTS:
        item = m.compact_npz(path)
        item["source"]["path"] = str(m.TRAIN_ROOTS[variant] / "diagnostics/step_00001.npz")
        bundle["arms"][variant]["positions"] = [item]
    report = m.analyze(bundle, allow_partial=True)
    before = copy.deepcopy(report)
    captured = {}

    def capture(fig, handle, **kwargs):
        fig.canvas.draw()
        captured[handle.name.rsplit("/", 1)[-1]] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    stems = ["qwen4_accuracy", "qwen4_pass", "qwen4_position_student_entropy"]
    m.plot_figures(report, tmp_path, only=stems)
    assert report == before
    assert set(captured) == {f"{stem}.{suffix}" for stem in stems for suffix in ("pdf", "png")}
    for stem, metric in (("qwen4_accuracy", "avg_at_8"), ("qwen4_pass", "pass_at_8")):
        fig = captured[f"{stem}.pdf"]
        assert fig.get_size_inches() == pytest.approx((5.5, 3.6))
        assert len(fig.axes) == 4
        for task, ax in zip(m.TASKS, fig.axes):
            assert ax.get_title() == m.TASK_LABELS[task]
            finite_values = [
                r[metric] * 100 for r in report["rows"]
                if r["task"] == task and r[metric] is not None
            ]
            low, high = ax.get_ylim()
            assert 0 <= low < high <= 100
            assert low <= min(finite_values) <= max(finite_values) <= high
            assert high - low >= min(10, max(finite_values) - min(finite_values))
            assert high - low < 100
            for variant, line in zip(m.VARIANTS, ax.lines):
                expected = [
                    r[metric] * 100 if r[metric] is not None else np.nan
                    for r in sorted(report["rows"], key=lambda r: r["step"])
                    if r["task"] == task and r["variant"] == variant
                ]
                np.testing.assert_equal(line.get_xdata(), sorted(m.STEPS))
                np.testing.assert_equal(line.get_ydata(), expected)
    heatmap = captured["qwen4_position_student_entropy.pdf"]
    assert heatmap.get_size_inches() == pytest.approx((5.5, 5.0))
    assert heatmap.axes[0].get_position().y0 > heatmap.axes[1].get_position().y1
    for variant, ax in zip(m.VARIANTS, heatmap.axes):
        item = report["diagnostics"][variant]["positions"][0]
        expected = m.position_means(item, "student_entropy", min_count=8)
        actual = ax.images[0].get_array().filled(np.nan)
        np.testing.assert_equal(actual[0], expected)
        assert np.isnan(actual[1:]).all()
    assert heatmap.axes[0].images[0].get_clim() == heatmap.axes[1].images[0].get_clim()
    for stem in stems:
        fig = captured[f"{stem}.pdf"]
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                assert item.get_fontsize() >= 8
        bbox = fig.get_tightbbox(canvas.get_renderer())
        assert bbox.x0 >= 0 and bbox.y0 >= 0
        assert bbox.x1 <= fig.get_figwidth() and bbox.y1 <= fig.get_figheight()


def test_figures_command_adds_only_selected_files_and_preserves_existing(
    m, bundle, tmp_path, monkeypatch
):
    input_path = tmp_path / "bundle.json"
    input_path.write_text(json.dumps(bundle))
    figures = tmp_path / "figures"
    figures.mkdir()
    old = figures / "qwen4_scores.pdf"
    old.write_bytes(b"existing immutable partial figure")
    before = input_path.read_bytes()

    def render(report, output, *, min_count, only):
        assert report["complete"] and min_count == 8
        for stem in only:
            for suffix in ("pdf", "png"):
                (output / f"{stem}.{suffix}").write_bytes(b"test figure")

    monkeypatch.setattr(m, "plot_figures", render)
    args = [
        "figures",
        "--input",
        str(input_path),
        "--figures-dir",
        str(figures),
        "--only",
        "qwen4_accuracy",
        "qwen4_pass",
    ]
    m.main(args)
    assert old.read_bytes() == b"existing immutable partial figure"
    assert input_path.read_bytes() == before
    manifest = json.loads((figures / "qwen4_figures_provenance.json").read_text())
    assert manifest["input_sha256"] == m.sha256(input_path)
    assert set(manifest["output_sha256"]) == {
        f"{stem}.{suffix}" for stem in ("qwen4_accuracy", "qwen4_pass") for suffix in ("pdf", "png")
    }
    with pytest.raises(SystemExit) as error:
        m.main(args)
    assert error.value.code == 2


def test_coverage_is_bin_max_count_not_sum_or_entropy_and_never_thresholded(
    m, diagnostic_cache, tmp_path, monkeypatch
):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    for arm in diagnostic_cache["arms"].values():
        for item in arm["positions"]:
            item["bin_size"] = 128
            item["metrics"]["student_entropy"] = {
                "sum": [0.0] * 128,
                "squared_sum": [0.0] * 128,
                "valid_count": [12, 3, 64] + [0] * 125,
                "max_position_count": [8, 2, 32] + [0] * 125,
            }
    report = m.analyze(diagnostic_cache, allow_partial=True)
    before = copy.deepcopy(report)
    captured = {}

    def capture(fig, handle, **kwargs):
        captured[handle.name.rsplit("/", 1)[-1]] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    m.plot_figures(report, tmp_path, only=["qwen4_coverage"], min_count=32)
    assert set(captured) == {"qwen4_coverage.pdf", "qwen4_coverage.png"}
    fig = captured["qwen4_coverage.pdf"]
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    assert fig.get_size_inches() == pytest.approx((5.5, 5))
    title = fig._suptitle.get_text()
    assert "count" in title and "128-token bin" in title and "PARTIAL" not in title
    assert fig.axes[0].get_position().y0 > fig.axes[1].get_position().y1
    for variant, ax in zip(m.VARIANTS, fig.axes):
        assert ax.get_title() == m.LABELS[variant]
        im = ax.images[0]
        assert im.get_clim() == (0, 32)
        assert tuple(im.get_extent()) == (0, 16384, 0.5, 200.5)
        expected = np.tile([8, 2, 32] + [np.nan] * 125, (200, 1))
        np.testing.assert_equal(im.get_array().filled(np.nan), expected)
        np.testing.assert_allclose(im.cmap(np.ma.masked), [221 / 255] * 3 + [1])
        assert all(not np.allclose(im.cmap(im.norm(x)), im.cmap(np.ma.masked)) for x in (2, 8, 32))
    for item in fig.findobj(Text):
        if item.get_visible() and item.get_text():
            assert item.get_fontsize() >= 8
    bbox = fig.get_tightbbox(canvas.get_renderer())
    assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= 5.5 and bbox.y1 <= 5
    assert report == before


def test_coverage_requires_student_entropy_and_renders_only_requested_files(
    m, bundle, diagnostic_cache, tmp_path
):
    import matplotlib.image as mpimg

    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    out = tmp_path / "coverage"
    with pytest.raises(ValueError, match="figure"):
        m.render_figures(path, out, only=["qwen4_coverage"])
    assert not out.exists()
    path.write_text(json.dumps(diagnostic_cache))
    before = path.read_bytes()
    m.render_figures(path, out, only=["qwen4_coverage"], allow_partial=True)
    assert {p.name for p in out.iterdir()} == {
        "qwen4_coverage.pdf",
        "qwen4_coverage.png",
        "qwen4_figures_provenance.json",
    }
    pixels = mpimg.imread(out / "qwen4_coverage.png")
    assert pixels.shape[:2] == (1000, 1100) and pixels[..., :3].std() > 0.02
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        m.render_figures(path, out, only=["qwen4_coverage"], allow_partial=True)


@pytest.mark.parametrize("complete", [True, False])
def test_dense_training_figures_use_diagnostic_completeness_and_paper_fonts(
    m, diagnostic_cache, tmp_path, monkeypatch, complete
):
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    for arm in diagnostic_cache["arms"].values():
        for row in arm["scalars"]["records"]:
            row.update(
                {
                    "diagnostics/teacher_entropy": 0.1,
                    "actor/grad_norm": 2.0,
                    "response_length/clip_ratio": 0.25,
                }
            )
    if not complete:
        diagnostic_cache["arms"]["token_opd"]["positions"].pop()
        diagnostic_cache["arms"]["token_opd"]["scalars"]["records"].pop()
    report = m.analyze(diagnostic_cache, allow_partial=True)
    assert not report["complete"]
    before = copy.deepcopy(report)
    captured = {}

    def capture(fig, handle, **kwargs):
        captured[handle.name.rsplit("/", 1)[-1]] = fig

    monkeypatch.setattr(Figure, "savefig", capture)
    m.plot_figures(report, tmp_path, only=["qwen4_training", "qwen4_position_student_entropy"])
    training = captured["qwen4_training.pdf"]
    assert training.get_size_inches() == pytest.approx((5.5, 5.5))
    assert len(training.axes) == 3
    for ax in training.axes:
        for line in ax.lines:
            np.testing.assert_equal(line.get_xdata(), range(1, 201))
            assert len(line.get_ydata()) == 200
    np.testing.assert_equal(training.axes[0].lines[0].get_ydata(), [0.2] * 200)
    np.testing.assert_equal(training.axes[0].lines[1].get_ydata(), [0.1] * 200)
    np.testing.assert_equal(training.axes[1].lines[0].get_ydata(), [2.0] * 200)
    np.testing.assert_equal(training.axes[2].lines[0].get_ydata(), [0.25] * 200)
    for stem in ("qwen4_training", "qwen4_position_student_entropy"):
        fig = captured[f"{stem}.pdf"]
        assert ("PARTIAL" not in fig._suptitle.get_text()) == complete
        for item in fig.findobj(Text):
            if item.get_visible() and item.get_text():
                assert item.get_fontsize() >= 8
    assert report == before


def test_build_cli_defaults_to_paper_figures(m, tmp_path, monkeypatch):
    captured = []

    def build(input_path, generated_dir, figures_dir, **kwargs):
        captured.append(str(figures_dir))
        return {"complete": True, "missing_checkpoints": []}

    monkeypatch.setattr(m, "build", build)
    m.main(["build", "--input", str(tmp_path / "bundle.json"), "--no-figures"])
    assert captured == ["paper/iclr2027/figures"]


def test_figure_selection_fail_closed_and_preflights_all_destinations(m, bundle, tmp_path):
    input_path = tmp_path / "bundle.json"
    input_path.write_text(json.dumps(bundle))
    out = tmp_path / "figs"
    with pytest.raises(ValueError, match="figure"):
        m.render_figures(input_path, out, only=["../study_overview"])
    assert not out.exists()
    with pytest.raises(ValueError, match="figure"):
        m.render_figures(input_path, out, only=["qwen4_training"])
    assert not out.exists()
    out.mkdir()
    (out / "qwen4_pass.png").write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError):
        m.render_figures(input_path, out, only=["qwen4_accuracy", "qwen4_pass"])
    assert sorted(p.name for p in out.iterdir()) == ["qwen4_pass.png"]
    del bundle["arms"]["token_opd"]["checkpoints"]["50"]
    input_path.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="missing checkpoints"):
        m.render_figures(input_path, out, only=["qwen4_accuracy"])


def test_step200_main_table_has_accepted_scores_deltas_and_paired_ci(m, bundle, tmp_path):
    add_questions(m, bundle, "block3_mean")
    token_rows = add_questions(m, bundle, "token_opd")
    for row in token_rows:
        row["correct_count"] = 2
    cp = bundle["arms"]["token_opd"]["checkpoints"]["200"]
    for task in m.TASKS:
        cp["result"]["per_task"][task]["avg_at_8"] = 0.25
    del bundle["arms"]["token_opd"]["checkpoints"]["50"]
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(bundle))
    m.build(
        path,
        tmp_path / "out",
        tmp_path / "figs",
        allow_partial=True,
        make_figures=False,
        bootstrap_replicates=32,
    )
    for language in ("en", "zh"):
        table = (tmp_path / f"out/qwen4_step200_{language}.tex").read_text()
        assert table.count("MATH500") == 1
        assert "+25.00 [25.00, 25.00]" in table
        assert "50.00 & 25.00" in table
        assert "PARTIAL" in table and "95\\%" in table
        assert table.count(r"\label{tab:qwen4endpoint}") == 1
        assert table.index(r"\caption{") < table.index(r"\label{tab:qwen4endpoint}")
        assert table.index(r"\label{tab:qwen4endpoint}") < table.index(r"\begin{tabular}")


def test_table_layout_retains_existing_float_rounding(m, bundle):
    for variant, score in (("block3_mean", 3016 / 4000), ("token_opd", 3037 / 4000)):
        bundle["arms"][variant]["checkpoints"]["200"]["result"]["per_task"]["math500"][
            "avg_at_8"
        ] = score
    report = m.analyze(bundle)
    before = copy.deepcopy(report)
    for language in ("en", "zh"):
        assert "MATH500 & Token OPD & 200 & 75.92" in m.tex_table(report, language)
        assert "MATH500 & 75.40 & 75.92 & -0.52 [NA]" in m.step200_table(report, language)
    assert report == before


def test_step200_table_does_not_fabricate_missing_bounds(m, bundle):
    table = m.step200_table(m.analyze(bundle), "en")
    assert "+0.00 [NA]" in table
    del bundle["arms"]["token_opd"]["checkpoints"]["200"]
    table = m.step200_table(m.analyze(bundle, allow_partial=True), "en")
    assert "50.00 & NA & NA" in table


def test_export_bundle_reads_local_mirror_and_preserves_original_source_hash(m, bundle, tmp_path):
    artifact_root = tmp_path / "mirror"
    for variant, arm in bundle["arms"].items():
        for step, cp in arm["checkpoints"].items():
            if variant == "token_opd" and step != "200":
                continue
            for name, value in (
                ("acceptance.json", cp["result"]),
                ("eval_card.json", {k: v for k, v in cp["eval_card"].items() if k != "source"}),
            ):
                original = m.EVAL_ROOTS[variant] / f"evaluations/{variant}_step{step}/{name}"
                local = artifact_root / original.relative_to(m.ROOT)
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(json.dumps(value))
                (local.parent / "exit_code.txt").write_text("0\n")
    exported = m.export_bundle(artifact_root, include_diagnostics=False, question_steps=())
    assert list(exported["arms"]["token_opd"]["checkpoints"]) == ["200"]
    cp = exported["arms"]["block3_mean"]["checkpoints"]["200"]
    local = artifact_root / m.EVAL_ROOTS["block3_mean"].relative_to(m.ROOT)
    assert cp["source"]["sha256"] == m.sha256(
        local / "evaluations/block3_mean_step200/acceptance.json"
    )
    assert (
        cp["source"]["path"]
        == bundle["arms"]["block3_mean"]["checkpoints"]["200"]["source"]["path"]
    )
    assert m.analyze(exported, allow_partial=True)["missing_checkpoints"] == [
        "token_opd:150",
        "token_opd:100",
        "token_opd:50",
    ]


def test_export_defaults_to_all_200_position_steps(m, monkeypatch, capsys):
    assert list(m.export_bundle.__kwdefaults__["position_steps"]) == list(range(1, 201))
    captured = []

    def export(root, **kwargs):
        captured.append(kwargs)
        return {}

    monkeypatch.setattr(m, "export_bundle", export)
    m.main(["export"])
    assert captured[0]["position_steps"] == list(range(1, 201))
    assert captured[0]["include_diagnostics"]
    assert json.loads(capsys.readouterr().out) == {}


@pytest.fixture
def diagnostic_cache(m, bundle, tmp_path):
    cache = copy.deepcopy(bundle)
    cache["arms"]["token_opd"]["checkpoints"] = {
        "200": cache["arms"]["token_opd"]["checkpoints"]["200"]
    }
    npz = tmp_path / "diagnostic.npz"
    make_npz(npz)
    item = m.compact_npz(npz, bin_size=16384)
    for variant, arm in cache["arms"].items():
        arm["positions"] = []
        for step in range(1, 201):
            value = copy.deepcopy(item)
            value["step"] = step
            value["source"]["path"] = str(
                m.TRAIN_ROOTS[variant] / f"diagnostics/step_{step:05d}.npz"
            )
            arm["positions"].append(value)
        arm["scalars"] = {
            "source": source(m.TRAIN_ROOTS[variant] / "diagnostics/scalars.jsonl"),
            "records": [
                {"step": step, "diagnostics/student_entropy": 0.2} for step in range(1, 201)
            ],
        }
    return cache


def test_merge_diagnostics_reuses_all_400_steps_without_replacing_new_results(
    m, bundle, diagnostic_cache, tmp_path, capsys
):
    evaluation, cache_path = tmp_path / "eval.json", tmp_path / "cache.json"
    evaluation.write_text(json.dumps(bundle))
    cache_path.write_text(json.dumps(diagnostic_cache))
    original = [path.read_bytes() for path in (evaluation, cache_path)]
    m.main(
        [
            "merge-diagnostics",
            "--input",
            str(evaluation),
            "--diagnostics-from",
            str(cache_path),
        ]
    )
    merged = json.loads(capsys.readouterr().out)
    assert m.analyze(merged)["complete"]
    for variant in m.VARIANTS:
        assert merged["arms"][variant]["checkpoints"] == bundle["arms"][variant]["checkpoints"]
        for key in ("scalars", "positions"):
            assert merged["arms"][variant][key] == diagnostic_cache["arms"][variant][key]
    assert merged["diagnostics_reuse"]["cache_sha256"] == m.sha256(cache_path)
    assert merged["diagnostics_reuse"]["evaluation_sha256"] == m.sha256(evaluation)
    assert [path.read_bytes() for path in (evaluation, cache_path)] == original


@pytest.mark.parametrize("issue", ["missing_position", "missing_scalar", "wrong_data", "conflict"])
def test_merge_diagnostics_fails_closed(m, bundle, diagnostic_cache, tmp_path, issue):
    if issue == "missing_position":
        diagnostic_cache["arms"]["token_opd"]["positions"].pop()
    elif issue == "missing_scalar":
        diagnostic_cache["arms"]["token_opd"]["scalars"]["records"].pop()
    elif issue == "wrong_data":
        diagnostic_cache["data_sha256"]["math500"] = "d" * 64
        for arm in diagnostic_cache["arms"].values():
            for cp in arm["checkpoints"].values():
                cp["result"]["sha256"][str(m.DATA / "math500.jsonl")] = "d" * 64
    else:
        bundle["arms"]["token_opd"]["scalars"] = copy.deepcopy(
            diagnostic_cache["arms"]["token_opd"]["scalars"]
        )
        bundle["arms"]["token_opd"]["scalars"]["records"][0]["diagnostics/student_entropy"] = 0.3
    evaluation, cache_path = tmp_path / "eval.json", tmp_path / "cache.json"
    evaluation.write_text(json.dumps(bundle))
    cache_path.write_text(json.dumps(diagnostic_cache))
    with pytest.raises(ValueError, match="diagnostic"):
        m.merge_diagnostics(evaluation, cache_path)
    if issue.startswith("missing_"):
        merged = m.merge_diagnostics(evaluation, cache_path, allow_partial_diagnostics=True)
        assert m.analyze(merged)["complete"]


def test_compact_question_export_checks_identity_seeds_and_hash(m, tmp_path):
    path = tmp_path / "math500_graded.jsonl"
    rows = [
        {
            "example_id": str(i),
            "rollout_id": j,
            "seed": 21 + j,
            "correct": j < 4,
            "response": "\\boxed{1}" if j else "missing",
            "finish_reason": "length" if j == 7 else "stop",
        }
        for i in range(500)
        for j in range(8)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    compact = m.compact_question_scores(path, "math500", m.sha256(path))
    assert len(compact) == 500
    assert compact[0]["correct_count"] == 4
    assert compact[0]["missing_box_count"] == compact[0]["length_stop_count"] == 1
    assert "response" not in compact[0]
    with pytest.raises(ValueError, match="hash"):
        m.compact_question_scores(path, "math500", "f" * 64)
    rows[0]["seed"] = 22
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="seed"):
        m.compact_question_scores(path, "math500", m.sha256(path))
