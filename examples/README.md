# Examples

Four short notebooks, meant to be read in order. Each is self-contained and
runs in seconds on a laptop — none of them reproduces a paper figure, that is
what [`REPRODUCING.md`](../REPRODUCING.md) is for. The point here is to make
the code readable.

```bash
pip install -r ../requirements.txt
jupyter lab
```

They are stored with their outputs, so you can also just read them on GitHub.

### [`01_max3cut_and_hamiltonians.ipynb`](01_max3cut_and_hamiltonians.ipynb)

The problem, before any quantum algorithm. How a ternary variable becomes a
qutrit level, what the cost Hamiltonian charges per edge, and why the mixer is
that particular combination. Ends by solving a six-vertex graph twice — once
from the spectrum, once by brute force over all $3^6$ colourings — and checking
the two agree.

### [`02_operator_pool.ipynb`](02_operator_pool.ipynb)

Where the operators come from: nested commutators of the adiabatic
Hamiltonian, truncated at order $\ell$. Shows the one genuine difference from
the qubit case — the commutators produce non-Hermitian products on a single
site, which is why the pool is Hermitized — and why a pool 25 times larger does
not mean a circuit 25 times deeper.

### [`03_running_qudit_adapt.ipynb`](03_running_qudit_adapt.ipynb)

A complete run on $G_1$. Which operator is selected at each step, why the warm
start makes the energy monotone, and a closing observation worth knowing: at
21 parameters the energy is still $6.6\times10^{-3}$ off, but the most probable
measurement outcome is already an exactly optimal colouring.

### [`04_native_gate_count.ipynb`](04_native_gate_count.ipynb)

Appendix B. Compiles the ansatz into the trapped-ion native gate set with the
decomposition algorithm of Ringbauer *et al.*, and reproduces Tables II–IV.
Includes the two facts that are easy to get wrong: diagonal generators are
*not* free (three pulses per relative phase, no virtual Z), and $\lambda_4$,
$\lambda_5$ cost extra because they live on a non-adjacent level pair.
