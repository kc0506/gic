# coding=utf-8
"""Unit test: gradient of ti.math.distance at d=0 and d~1e-8 (CPU, no GPU).

Mirrors gic's compute_loss_* usage: loss += distance(x[p], gt) with autodiff.
"""
import taichi as ti

ti.init(arch=ti.cpu, debug=False)

N = 4
x = ti.Vector.field(3, ti.f32, shape=N, needs_grad=True)
gt = ti.Vector.field(3, ti.f32, shape=N)
loss = ti.field(ti.f32, shape=(), needs_grad=True)


@ti.kernel
def compute_loss():
    for i in range(N):
        loss[None] += ti.math.distance(x[i], gt[i])


def run_case(name: str, offsets: list) -> None:
    """offsets: per-point displacement of gt from x."""
    for i in range(N):
        x[i] = [0.1 * (i + 1), 0.2, 0.3]
        gt[i] = [0.1 * (i + 1) + offsets[i][0], 0.2 + offsets[i][1], 0.3 + offsets[i][2]]
    x.grad.fill(0)
    loss[None] = 0.0
    loss.grad[None] = 1.0
    compute_loss()
    compute_loss.grad()
    grads = [x.grad[i].to_numpy() for i in range(N)]
    print(f"[{name}] loss={loss[None]:.6e}")
    for i, g in enumerate(grads):
        print(f"  point {i}: offset={offsets[i]}, grad={g}")


run_case("all far (control)", [[0.01, 0.0, 0.0]] * 4)
run_case("one EXACT zero", [[0.0, 0.0, 0.0], [0.01, 0, 0], [0.01, 0, 0], [0.01, 0, 0]])
run_case("one tiny 1e-8", [[1e-8, 0.0, 0.0], [0.01, 0, 0], [0.01, 0, 0], [0.01, 0, 0]])
run_case("all EXACT zero", [[0.0, 0.0, 0.0]] * 4)
