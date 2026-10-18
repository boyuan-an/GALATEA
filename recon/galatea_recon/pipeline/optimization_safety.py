import json
import math
import os
from pathlib import Path

import torch


REPORT_VERSION = 1


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".%s.tmp" % os.getpid())
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


class UnsafeOptimizationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "%s has no finite evaluated state (%s); refusing to export a trajectory"
            % (report["stage"], report["stop_reason"])
        )


def _all_finite(tensors):
    tensors = list(tensors)
    return not tensors or bool(torch.stack([torch.isfinite(t).all() for t in tensors]).all())


def optimize_stage(
    name, optimizer, loss_fn, max_iterations, *, stopper=None, post_step=None, print_every=100
):
    if max_iterations < 0 or print_every < 1:
        raise ValueError("max_iterations must be >= 0 and print_every must be >= 1")
    params = [p for group in optimizer.param_groups for p in group["params"]]
    best_params = None
    report = {
        "stage": name,
        "enabled": stopper is not None,
        "max_iterations": max_iterations,
        "iterations_run": 0,
        "initial_loss": None,
        "last_evaluated_loss": None,
        "best_loss": None,
        "best_step": None,
        "last_evaluated_step": None,
        "stop_reason": "iteration_limit",
        "stopped_early": False,
        "restored_best": False,
        "recovered_nonfinite": False,
        "last_checkpoint": None,
        "checkpoints": [],
    }
    for completed in range(max_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        if not _all_finite(params):
            report["stop_reason"] = "nonfinite_parameters"
            break
        with torch.set_grad_enabled(completed < max_iterations):
            total, components = loss_fn()
        value = float(total.detach())
        report["last_evaluated_step"] = completed
        report["last_evaluated_loss"] = value if math.isfinite(value) else None
        if not math.isfinite(value):
            report["stop_reason"] = "nonfinite_loss"
            break
        if completed == 0:
            report["initial_loss"] = value
        if best_params is None or value < report["best_loss"]:
            best_params = [p.detach().clone() for p in params]
            report["best_loss"] = value
            report["best_step"] = completed

        decision = None
        if stopper is not None and completed > 0:
            metrics = {"total": value}
            if "contact" in components:
                metrics["contact"] = float(components["contact"].detach())
            decision = stopper.observe(completed, metrics)
            if decision is not None:
                report["checkpoints"].append(decision)
                report["last_checkpoint"] = decision
        stop = bool(decision and decision["stop"] and completed < max_iterations)
        if completed % print_every == 0 or completed == max_iterations or stop:
            details = " ".join(
                "%s=%.6f" % (key, float(component.detach()))
                for key, component in components.items()
                if float(component.detach()) != 0
            )
            print(
                "  [%s] step %4d | total=%.6f | best=%.6f at step %d | %s"
                % (name, completed, value, report["best_loss"], report["best_step"], details)
            )
        if stop:
            report["stop_reason"] = "early_stop"
            break
        if completed == max_iterations:
            break
        total.backward()
        if not _all_finite(p.grad for p in params if p.grad is not None):
            report["stop_reason"] = "nonfinite_gradient"
            break
        optimizer.step()
        report["iterations_run"] = completed + 1
        if post_step is not None:
            with torch.no_grad():
                post_step()

    report["recovered_nonfinite"] = report["stop_reason"].startswith("nonfinite_")
    report["stopped_early"] = report["iterations_run"] < max_iterations
    if best_params is None:
        raise UnsafeOptimizationError(report)
    report["restored_best"] = (
        report["best_step"] != report["iterations_run"] or report["recovered_nonfinite"]
    )
    with torch.no_grad():
        for parameter, best in zip(params, best_params):
            parameter.copy_(best)
    optimizer.zero_grad(set_to_none=True)
    print(
        "  [%s] export step %d/%d, loss %.6f (%s)"
        % (
            name,
            report["best_step"],
            report["iterations_run"],
            report["best_loss"],
            report["stop_reason"],
        )
    )
    return report
