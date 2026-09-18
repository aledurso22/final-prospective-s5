import importlib.util
import subprocess
import sys

import jax.numpy as jnp
import numpy as np

from s5.ssm import apply_ssm, discretize_zoh
from s5.ssm_init import make_DPLR_HiPPO


UPSTREAM_COMMIT = "3c18fdb6b06414da35e77b94b9cd855f6a95ef17"


def _git_show(path):
    return subprocess.check_output(
        ["git", "show", f"{UPSTREAM_COMMIT}:{path}"], text=True)


def _upstream_ssm():
    package_name = "pinned_upstream_s5"
    root = __import__("tempfile").mkdtemp(prefix="s5-upstream-")
    package_dir = f"{root}/{package_name}"
    __import__("os").makedirs(package_dir)
    sys.path.insert(0, root)
    open(f"{package_dir}/__init__.py", "w").close()
    with open(f"{package_dir}/ssm_init.py", "w") as handle:
        handle.write(_git_show("s5/ssm_init.py"))
    module_path = f"{package_dir}/ssm.py"
    with open(module_path, "w") as handle:
        # The pinned source predates the removal of jax.numpy.DeviceArray.
        # This changes annotations only; executable recurrence code is intact.
        handle.write(_git_show("s5/ssm.py").replace(
            "np.DeviceArray", "jax.Array"))
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.ssm", module_path,
        submodule_search_locations=[])
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package_name
    sys.modules[package_name] = importlib.import_module(package_name)
    sys.modules[f"{package_name}.ssm"] = module
    spec.loader.exec_module(module)
    return module


def test_native_discretization_matches_pinned_upstream_numerically():
    upstream = _upstream_ssm()
    lam = jnp.asarray([-0.4 + 0.2j, -1.1 - 0.3j], dtype=jnp.complex128)
    b = jnp.asarray([[0.3 + 0.1j], [-0.2 + 0.4j]], dtype=jnp.complex128)
    delta = jnp.asarray([0.7, 0.2], dtype=jnp.float64)
    local = discretize_zoh(lam, b, delta)
    reference = upstream.discretize_zoh(lam, b, delta)
    np.testing.assert_allclose(np.asarray(local[0]), np.asarray(reference[0]))
    np.testing.assert_allclose(np.asarray(local[1]), np.asarray(reference[1]))


def test_native_scan_matches_pinned_upstream_numerically():
    upstream = _upstream_ssm()
    lam = jnp.asarray([0.8 + 0.1j, 0.6 - 0.2j], dtype=jnp.complex128)
    b = jnp.asarray([[0.2 + 0.1j], [0.1 - 0.3j]], dtype=jnp.complex128)
    c = jnp.asarray([[0.4 - 0.2j, -0.1 + 0.3j]], dtype=jnp.complex128)
    x = jnp.asarray([[0.2], [-0.4], [0.7]], dtype=jnp.float64)
    local = apply_ssm(lam, b, c, x, False, False)
    reference = upstream.apply_ssm(lam, b, c, x, False, False)
    np.testing.assert_allclose(np.asarray(local), np.asarray(reference))


def test_native_module_output_matches_pinned_upstream():
    upstream = _upstream_ssm()
    Lambda, _, _, V, _ = make_DPLR_HiPPO(4)
    V = V[:, :2]
    kwargs = dict(H=3, P=2, Lambda_re_init=Lambda[:2].real,
                  Lambda_im_init=Lambda[:2].imag, V=V, Vinv=V.conj().T,
                  C_init="lecun_normal", discretization="zoh",
                  dt_min=0.001, dt_max=0.1, conj_sym=True,
                  clip_eigs=True, bidirectional=False)
    from s5.ssm import S5SSM
    local = S5SSM(**kwargs)
    reference = upstream.S5SSM(**kwargs)
    x = jnp.asarray([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]], dtype=jnp.float64)
    key = __import__("jax").random.PRNGKey(19)
    local_variables = local.init({"params": key}, x)
    reference_variables = reference.init({"params": key}, x)
    np.testing.assert_allclose(
        np.asarray(local.apply(local_variables, x)),
        np.asarray(reference.apply(reference_variables, x)),
        rtol=1e-6, atol=1e-6)
