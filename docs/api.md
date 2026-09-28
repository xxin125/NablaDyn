# NablaDyn API

A guide to the functions and classes you use, in the order you meet them. Each entry is the
docstring of NablaDyn 0.1.0, so it matches this version. Less common helpers are documented in their
docstrings: `help(nabladyn.<module>)`.

## Neighbor lists

Every model takes `pairs`, the atom pairs within reach of its cutoff, from a `NeighborList`:
`allocate` sizes it for a frame and builds it, `update` reuses it for the next frames until a
rebuild is needed, and `tensors()` returns the pairs. MD builds and updates its own list; build one
yourself for single frames, analysis and your own loops.

### neighbors.NeighborList

```python
neighbors.NeighborList(cutoff: Real, *, skin: Real = 0.0, capacity_multiplier: Real = 1.25)
```

Pairs of atoms closer than the list radius, cutoff + skin, in nm.

`allocate` sizes the list to `capacity_multiplier` times a first frame's pair count
and builds it. `update` reuses the pairs until an atom has moved far enough that a pair
within the cutoff could be missing, then rebuilds. `tensors()` returns the pairs. Cells
are orthorhombic and periodic; positions may lie outside the cell.

Raises `ValueError` unless `cutoff` is positive, `skin` nonnegative and
`capacity_multiplier` at least 1, all finite.

#### NeighborList.allocate

```python
allocate(positions: Tensor, cell: Tensor) -> None
```

Count the pairs of positions [N, 3] in the cell [3], size the list and build it.

Positions and cell are CUDA float32 or float64 tensors in nm. It also recovers a
list that failed an update. Syncs with the host.

Raises `TypeError` or `ValueError` for invalid positions or cell, and
`ValueError` when the grid or the pairs do not fit int32.

#### NeighborList.update

```python
update(positions: Tensor, cell: Tensor) -> None
```

Reuse the pairs for positions [N, 3] in the cell [3], or rebuild them.

The pairs are reused while no atom has moved more than (R s_min - cutoff) / 2 from its
position at the last rebuild, scaled with the cell; R is the list radius and s_min the
smallest ratio of a cell length to its length then. In a fixed cell that is half the
skin, so zero skin rebuilds every time. Syncs with the host.

Raises `TypeError` or `ValueError` for positions or a cell of the wrong kind or
unlike the allocated ones. Raises `ValueError`, and fails the list until
`allocate`, for a cell length not finite or not above twice the cutoff, a coordinate
that is not finite, or more pairs than the capacity. Raises `RuntimeError` before
`allocate` or after a failure.

#### NeighborList.tensors

```python
tensors() -> NeighborPairs
```

Return the pairs of the last build, the `NeighborPairs` that models take.

Raises `RuntimeError` before `allocate` or after a failure.

#### NeighborList.num_pairs

```python
property num_pairs: int
```

The number of pairs of the last build.

Syncs with the host. Raises `RuntimeError` before `allocate` or after a failure.

## Build a model

Each constructor returns a model: a PyTorch module with one scalar energy in kJ/mol.
`model(positions, cell, pairs)` gives the energy of a frame, with `pairs` from a `NeighborList` at
the model's cutoff, and `quantity` gives forces and virial. A model computes in the dtype and on the
device of its parameters. `lennard_jones`, `schnet` and `tabulated` take `trainable`, the parameters
that `training.fit` may change.

### potential.lennard_jones

```python
potential.lennard_jones(
    *,
    atom_types: Tensor,
    sigma: Tensor,
    epsilon: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential
```

Build a 12-6 Lennard-Jones model shifted to zero at the cutoff.

E sums u(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] - u_c over the pairs closer than the
cutoff rc, with u_c = 4 eps [(sigma/rc)^12 - (sigma/rc)^6] of each pair's own sigma and
epsilon. E is continuous at rc and is the energy MD conserves and samples. u_c is
constant, so forces and virial are those of the unshifted potential; the force steps
to zero at rc.

`atom_types` [N], int64 in [0, T), fixes the atoms. `sigma` (nm) and `epsilon`
(kJ/mol) hold T(T+1)/2 values each, one per unordered type pair in the order (0, 0),
(0, 1), ..., (1, 1), ...; no mixing rule is applied. `cutoff` is in nm. `trainable`
names what a fit trains, among "sigma" and "epsilon": a tuple trains every row of each;
a mapping trains only the listed rows, as {"epsilon": [1]} for (0, 1), and holds the
rest. Only named parameters require gradients. The model keeps copies of its inputs and
computes in the dtype and on the device of `sigma`. Syncs with the host.

Raises `ValueError` for a `trainable` that names other parameters or rows outside
[0, T(T+1)/2), a cutoff that is not finite and positive, parameters that are not finite
positive float vectors of T(T+1)/2 values, or atom types outside [0, T).

### potential.opls

```python
potential.opls(
    *,
    topology: Topology,
    parameters: Parameters,
    cutoff: Real,
    cell: Tensor,
    ewald_tolerance: Real = 5e-4,
) -> Potential
```

Build an OPLS-AA model of a fixed topology, with Coulomb by smooth particle-mesh Ewald.

U = sum_bonds 0.5 k (r - r0)^2 + sum_angles 0.5 k (theta - theta0)^2 +
sum_dihedrals k [1 + cos(n phi - phase)] + U_LJ + U_C, each term with its type's row of
`parameters`. theta = atan2(|a x b|, a . b) with a = r_i - r_j and b = r_k - r_j; phi
is the IUPAC dihedral of i-j-k-l in (-pi, pi], 0 cis and pi trans. Every vector is the
minimum image, so a molecule must span less than half of each cell length.

`topology.bonds` sets the pair classes. Pairs 1 or 2 bonds apart do not interact; 1-4
pairs, 3 bonds apart, count 0.5 u(r) + 0.5 C q_i q_j / r at any distance, unshifted.
Other pairs within the cutoff count u(r) - u(rc) and the real-space Ewald term
C q_i q_j [erfc(alpha r)/r - erfc(alpha rc)/rc], both zero at rc; an order-5 smooth PME
mesh adds every pair's long-range part, less the self and net-charge terms. Here
u(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] with sigma = sqrt(sigma_a sigma_b) and
eps = sqrt(eps_a eps_b) of the two atoms' LJ types; a zero product gives no term, and a
zero sigma or epsilon gets zero gradient. C is `units.COULOMB`; q are the charges.

`cell` [3], in nm, plans the Ewald splitting once for `ewald_tolerance` delta, about
the relative force error: alpha = sqrt(-ln(2 delta)) / cutoff, and about
2 alpha L / (3 delta^(1/5)) mesh points per axis, rounded up to a product of 2, 3 and 5.
Both stay fixed as `model.alpha` (nm^-1) and `model.mesh`, so accuracy drops silently
in a larger cell: plan with the largest cell a run visits.

`topology` is an `opls.Topology` and `parameters` an `opls.Parameters`; their
docstrings give fields, shapes and units. `cutoff` (nm) holds for both the LJ and the
real-space terms. The model keeps copies of the tables as frozen parameters and of the
topology as buffers, and computes in the dtype and on the device of `parameters`.
Syncs with the host.

Raises `ValueError` for a `topology` or `parameters` of another type, a cutoff that
is not finite and positive, a cell that is not three finite positive lengths, a tolerance
outside (0, 0.5), tables that are not finite vectors of one float dtype and device,
paired tables of different lengths, topology fields of another dtype or shape, atoms or
types out of range, sigma or epsilon below 0, or periodicities below 1.

### potential.opls.Topology

```python
potential.opls.Topology(
    *,
    charge_types: Tensor,
    lj_types: Tensor,
    bonds: Tensor,
    bond_types: Tensor,
    angles: Tensor,
    angle_types: Tensor,
    dihedrals: Tensor,
    dihedral_types: Tensor,
)
```

Which atoms interact, and which parameter row each atom and each interaction reads.

All fields are int64 tensors, for N atoms, B bonds, A angles and D dihedrals:

- `charge_types` [N]: each atom's row of `Parameters.charge`;
- `lj_types` [N]: each atom's row of `Parameters.sigma` and `Parameters.epsilon`;
- `bonds` [B, 2] and `bond_types` [B]: each bond's row of `bond_r0` and `bond_k`;
- `angles` [A, 3], the vertex in the middle, and `angle_types` [A];
- `dihedrals` [D, 4], propers and impropers in the force field's atom order, and
  `dihedral_types` [D]; a dihedral with several Fourier terms takes one row per term.

The bonds fix the pair classes by the shortest bond path: atoms 1 or 2 bonds apart do
not interact; atoms 3 bonds apart are 1-4 pairs, scaled by 0.5.

### potential.opls.Parameters

```python
potential.opls.Parameters(
    *,
    charge: Tensor,
    sigma: Tensor,
    epsilon: Tensor,
    bond_r0: Tensor,
    bond_k: Tensor,
    angle_theta0: Tensor,
    angle_k: Tensor,
    dihedral_k: Tensor,
    dihedral_periodicity: Tensor,
    dihedral_phase: Tensor,
)
```

The force field's parameter tables, one row per type of the `Topology`.

Vectors of one float dtype, float32 or float64, on one device, except
`dihedral_periodicity`, int64 on that device:

- `charge` [Tq] in e;
- `sigma` [Tlj] in nm and `epsilon` [Tlj] in kJ/mol, both >= 0, mixed geometrically
  for each pair; a type with epsilon 0 has no Lennard-Jones term;
- `bond_r0` [Tb] in nm and `bond_k` [Tb] in kJ/mol/nm^2: 0.5 k (r - r0)^2;
- `angle_theta0` [Ta] in rad and `angle_k` [Ta] in kJ/mol/rad^2: 0.5 k (theta - theta0)^2;
- `dihedral_k` [Td] in kJ/mol, `dihedral_periodicity` [Td], n >= 1, and
  `dihedral_phase` [Td] in rad: k [1 + cos(n phi - phase)].

### potential.schnet

```python
potential.schnet(
    *,
    atom_types: Tensor,
    num_types: int,
    cutoff: Real,
    features: int = 64,
    interactions: int = 3,
    gaussians: int = 50,
    trainable: bool = False,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> Potential
```

Build a SchNet network (Schütt et al., J. Chem. Phys. 148, 241722, 2018).

Each atom starts from its type's row of a table [T, F], x_i = table[t_i]. Each of the
`interactions` blocks adds ssp(m_i D1 + d1) D2 + d2, where m_i sums f_ij x_j A over
the pairs within the cutoff, each pair sending to both its atoms. The filter is
f_ij = (ssp(g(r_ij) B1 + b1) B2 + b2) C(r_ij), with K = `gaussians` fixed Gaussians
g_k(r) = exp(-(r - k D)^2 / (2 D^2)), D = cutoff / (K - 1), and the envelope
C(r) = (1 + cos(pi r / cutoff)) / 2, so a pair's contribution to the energy and forces
vanishes smoothly at the cutoff. The energy is E = sum_i ssp(x_i R1 + e1) R2 + e2 in
kJ/mol, with ssp(x) = ln(1 + e^x) - ln 2.

`atom_types` [N], int64 in [0, `num_types`), fixes the atoms and the device;
`cutoff` is in nm. `features` F, even, sizes the table, filters and blocks; the
readout's hidden layer is F / 2. `trainable` gives every weight a gradient, or none.
Weights are drawn in float64 from `seed` (table standard normal, matrices
Xavier-uniform, biases 0) and cast to `dtype`, so one seed gives one network in both
dtypes. The state dict holds only the weights, so fitted weights load into a network of
other atoms with the same types and sizes. Syncs with the host.

Raises `ValueError` for sizes that are not integers in range or an odd `features`,
a cutoff that is not finite and positive, a `dtype` other than float32 or float64,
or atom types outside [0, `num_types`).

### potential.tabulated

```python
potential.tabulated(
    *,
    atom_types: Tensor,
    table: Tensor,
    start: Tensor,
    cutoff: Real,
    trainable: tuple[str, ...] | Mapping[str, list[int]] = (),
) -> Potential
```

Build a pair potential tabulated per unordered type pair, 0 at the cutoff.

E sums u(r) over the pairs closer than the cutoff, each reading the row of its two
types. Between knots u is a C1 Catmull-Rom cubic, exact for any quadratic; below a
row's first knot u follows its tangent. Rows end at 0 at the cutoff, so the energy is
continuous there; the force steps to zero.

`atom_types` [N], int64 in [0, T), fixes the atoms. `table` [P, K], P = T(T+1)/2 and
K >= 4, holds u in kJ/mol at each row's knots, one row per unordered type pair in the
order (0, 0), (0, 1), ..., (1, 1), ...; its last column, at the cutoff, must be 0. Row
p's knots lie at start_p + k h_p, h_p = (cutoff - start_p) / (K - 1), with `start` [P]
and `cutoff` in nm. `trainable` is ("table",) to train every row or {"table": rows}
to train only those; no fit moves the last knot. The model computes in the dtype and on
the device of `table`. Syncs with the host.

Raises `ValueError` for a `trainable` that names another parameter or rows outside
[0, P), a cutoff that is not finite and positive, a table that is not a finite float
matrix of T(T+1)/2 rows and at least 4 columns ending in 0, a `start` that is not [P]
in the table's dtype and device or lies outside [0, cutoff), or atom types outside
[0, T).

## Run molecular dynamics

`maxwell_boltzmann` draws starting velocities; `nve`, `nvt` and `npt` build a `Simulation`, and
`Simulation.run` advances it. Each sample holds the panel, `md.FIELDS`: `potential_energy`,
`kinetic_energy`, `total_energy`, `temperature` and `pressure`, plus `conserved_energy` in NVT and
NPT and `volume` in NPT. `Simulation.state` copies the current frame as a `State`, from which a new
simulation continues a run; `read_sample_file` reads a sample file back.

### md.maxwell_boltzmann

```python
md.maxwell_boltzmann(masses: Tensor, temperature: Real, seed: Integral) -> Tensor
```

Draw velocities from the Maxwell-Boltzmann distribution at `temperature` in K.

Returns velocities [N, 3] in nm/ps for `masses` [N] in Da, in their dtype and on their
device, with zero total momentum and scaled to exactly `temperature` over 3N - 3
degrees of freedom. `seed` fixes the draw on any device. Syncs with the host.

Raises `ValueError` unless `masses` is a finite positive float32 or float64 vector
of at least 2 atoms, `temperature` a finite nonnegative number and `seed` an
integer.

### md.nve

```python
md.nve(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
) -> Simulation
```

Build a simulation at constant energy, by velocity Verlet.

`positions` [N, 3] in nm, which may lie outside the cell; `velocities` [N, 3] in
nm/ps; `masses` [N] in Da; `cell` [3] in nm; all in the model's dtype and on its
device. `timestep` is in ps. The neighbor list takes the model's cutoff, `skin`
in nm and `capacity_multiplier`. `images` [N, 3], int64, and `step` continue an
earlier run's count. The center-of-mass velocity is removed before the first step.
Syncs with the host.

Raises `TypeError` for a model that is not a `Potential`, and `ValueError` for
inputs of the wrong shape, dtype, device or value.

### md.nvt

```python
md.nvt(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    temperature: Real,
    thermostat_timescale: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
    thermostat: ChainState | None = None,
) -> Simulation
```

Build a simulation at constant temperature, by velocity Verlet and a Nose-Hoover chain.

Takes the arguments of `nve`, plus `temperature` in K, `thermostat_timescale` tau
in ps and, to continue a run, the `thermostat` state (float64). The 3 chain masses are
g kT tau^2, kT tau^2 and kT tau^2, g = 3N - 3. The panel of samples (`FIELDS`, see
`Simulation.run`) adds `conserved_energy`: U + K plus the chain's energy.

Raises `ValueError` unless the temperature and the timescale are finite and positive,
for fewer than 2 atoms or a `thermostat` that is not [3] float64 on the model's
device, and as `nve` does.

### md.npt

```python
md.npt(
    model: Potential,
    positions: Tensor,
    velocities: Tensor,
    masses: Tensor,
    cell: Tensor,
    *,
    timestep: Real,
    temperature: Real,
    pressure: Real,
    thermostat_timescale: Real,
    barostat_timescale: Real,
    skin: Real = 0.1,
    capacity_multiplier: Real = 1.25,
    images: Tensor | None = None,
    step: Integral = 0,
    thermostat: ChainState | None = None,
    barostat: BarostatState | None = None,
) -> Simulation
```

Build a simulation at constant pressure and temperature, by isotropic MTK.

MTK is Martyna-Tobias-Klein dynamics. Takes the arguments of `nvt`, plus `pressure`
in kJ/mol/nm^3 (`units.BAR` converts), `barostat_timescale` tau_b in ps and, to
continue a run, the `barostat` state. Nose-Hoover chains of 3 thermostats act on the
particles and on the barostat; the box mass is 3 (N + 1) kT tau_b^2. The cell keeps its
aspect ratio and scales about the origin.

The panel of samples (`FIELDS`, see `Simulation.run`) adds `volume` in nm^3 and
`conserved_energy`: U + K plus P V, the box's kinetic energy and both chains'
energies. Started far from `pressure`, the box oscillates, as with any extended-system
barostat.

Raises `ValueError` unless the pressure is finite and the temperature and timescales
finite and positive; for a `barostat` whose momentum is not [] float64 or whose chain
is not [3] float64 on the model's device; and as `nvt` does.

### md.Simulation

```python
class Simulation
```

Molecular dynamics of a potential on the GPU; build it with `nve`, `nvt` or `npt`.

It owns its frame, a neighbor list at the model's cutoff and a CUDA Graph of one step,
which `run` replays. A second graph, captured on first use, evaluates the panel of
`FIELDS`: potential, kinetic and total energy in kJ/mol, the temperature over 3N - 3
degrees of freedom in K, and the pressure in kJ/mol/nm^3. The step reads the model's
parameter storage, so a parameter changed in place reaches the next step.

#### Simulation.run

```python
run(
    steps: Integral,
    *,
    sample_every: Integral | None = None,
    echo: bool | TextIO = False,
    sample_file: str | os.PathLike | None = None,
) -> dict[str, Tensor] | None
```

Advance `steps` steps.

With `sample_every`, returns the panel at each step that is a multiple of it, as
host tensors [S] under the names of `FIELDS` and the ensemble's fields, and
`step` [S], int64; otherwise None. `echo` (True for stdout, or a text stream)
prints a header and one row per sample. `sample_file` names an HDF5 file,
overwritten, holding each sample's frame and panel (see `read_sample_file`). Both
need `sample_every`.

Syncs with the host: reads the neighbor list's status every 1,000 steps, before each
echoed or written sample, and at the end.

Raises `ValueError` for settings out of range, `RuntimeError` after a failure.
A neighbor list error raises `ValueError`, discards the call's samples and fails
the simulation; the file keeps the samples written before it.

#### Simulation.sample

```python
sample() -> dict[str, float]
```

Return the current frame's panel under the names of `FIELDS` and the ensemble's.

Syncs with the host. Raises `RuntimeError` after a failure.

#### Simulation.state

```python
state() -> State
```

Return a copy of the current frame as a `State`.

Raises `RuntimeError` after a failure.

### md.State

```python
md.State(
    positions: Tensor,
    velocities: Tensor,
    forces: Tensor,
    cell: Tensor,
    images: Tensor,
    step: int,
    thermostat: ChainState | None = None,
    barostat: BarostatState | None = None,
)
```

A copy of a simulation's frame, in nm, nm/ps and kJ/mol/nm.

`positions` [N, 3] lie in [0, L); `images` [N, 3], int64, count lattice crossings,
so the unwrapped positions are `positions + images * cell`. `forces` [N, 3] act at
`positions`; `cell` is [3]. `step` counts the steps since the first frame.
`thermostat` is the chain of an NVT or NPT run, `barostat` the barostat of an NPT
run. Changing the tensors changes no simulation. To continue a run, pass `positions`,
`velocities`, `cell`, `images` and `step`, with `thermostat` (NVT, NPT) and
`barostat` (NPT), to `nve`, `nvt` or `npt`.

### md.read_sample_file

```python
md.read_sample_file(path: str | os.PathLike) -> tuple[dict[str, np.ndarray], dict]
```

Read a sample file's samples and attributes.

Samples are NumPy arrays under their dataset names: `step`, `time`, the frame arrays
`positions`, `velocities`, `forces`, `cell` and `images`, `masses` and the
panel fields. Attributes are the ensemble and its settings, the timestep in ps and the
cutoff in nm. The file holds no thermostat or barostat state, so it is not a restart;
continue a run from `Simulation.state()`.

Raises `ValueError` for a file that is not a NablaDyn sample file.

## Train

A `Dataset` holds labeled frames: positions and cells with energies, forces or virials. `fit` trains
a model's trainable parameters on it with any `torch.optim` optimizer and returns a `Report`: the
loss of each update and the RMS errors before and after.

### training.Dataset

```python
training.Dataset(
    positions: np.ndarray | Tensor,
    cells: np.ndarray | Tensor,
    *,
    energy: np.ndarray | Tensor | None = None,
    forces: np.ndarray | Tensor | None = None,
    virial: np.ndarray | Tensor | None = None,
)
```

Labeled frames of one topology, copied to the CPU.

`positions` [F, N, 3] and `cells` [F, 3] in nm; optional labels `energy` [F] in
kJ/mol, `forces` [F, N, 3] in kJ/mol/nm and `virial` [F, 3, 3] in kJ/mol. All are
NumPy arrays or tensors of one dtype, float32 or float64. Indexing with an integer, a
slice or an integer array returns those frames as a new Dataset.

Raises `ValueError` for a wrong shape or dtype, a non-finite value or a nonpositive cell.

### training.fit

```python
training.fit(
    model: Potential,
    train: Dataset,
    optimizer: torch.optim.Optimizer,
    *,
    updates: Integral,
    batch_size: Integral,
    energy: Real | None = None,
    forces: Real | None = None,
    virial: Real | None = None,
    validation: Dataset | None = None,
    seed: Integral = 0,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    echo_every: Integral | None = None,
) -> Report
```

Fit the model's trained parameters to the training labels and return a `Report`.

Trained parameters are those that require gradients. A channel enters the loss when its
weight is given (`energy`, `forces`, `virial`), as weight x mean ((X - X*) / s)^2
with the scale s from the training labels. Each of the `updates` takes the next batch
of `batch_size` frames (each epoch a new shuffle from `seed`, its last batch the
remainder), writes the gradient of the batch's mean loss into `.grad` and calls
`optimizer.step()`. Rows the model does not train keep their values. Parameters change
in place; another call continues from them and the optimizer. `scheduler`, built on the
optimizer, steps after each update; one that needs a metric, such as
`ReduceLROnPlateau`, is not supported. The report gives the RMS errors before and after
on the training frames and on `validation`, which is not trained on.

With `echo_every` = k, prints a header, then every k updates the update number, the
mean loss and the mean unweighted terms over those k updates.

Syncs with the host.

Raises `TypeError` for a model that is not a Potential, and `ValueError` for weights
or counts out of range, a model that trains no parameter, or frames with another atom
count or without the weighted labels.

### training.Report

```python
training.Report(
    loss: np.ndarray,
    scales: dict[str, float],
    training: dict[str, tuple[float, float]],
    validation: dict[str, tuple[float, float]] | None,
)
```

The result of a fit.

`loss` [updates]: each update's batch loss, before its step. `scales`: the channel
scales from the training labels. `training` and `validation` (None without a
validation set): each weighted channel's RMS error (before, after) the fit, for the
energy per frame in kJ/mol, force components in kJ/mol/nm and virial components in kJ/mol.

## Forces and virials

`quantity` turns a model into a function of `(positions, cell, pairs)`. `energy_force_virial`
returns all three; `force`, `energy_and_force`, `virial`, `force_virial` and `energy_virial` return
only what they name. Forces and virial are exact autograd derivatives of the energy.

### quantity.energy_force_virial

```python
quantity.energy_force_virial(energy_fn: Callable[..., Tensor])
```

Bind (E, F, W): the energy and virial [3, 3] in kJ/mol, forces [N, 3] in kJ/mol/nm.

`energy_fn` is a model, or any energy with its call signature. The result is a
function of `(positions, cell, pairs)`, with `pairs` from a `NeighborList`, that
returns all three from one evaluation. F = -dE/dpositions; W = -dE/dstrain at zero
strain, with no kinetic term, so the pressure is (2K + trace W) / (3V).

## Analysis

`observables` computes instantaneous quantities of a frame: the kinetic energy, the temperature, the
pressure and `rdf`, the radial distribution function g(r) of each type pair.

### observables.rdf

```python
observables.rdf(
    positions: Tensor,
    cell: Tensor,
    pairs: NeighborPairs,
    edges: Tensor,
    types: Tensor | None = None,
) -> Tensor
```

Return the radial distribution function of one frame, [C, B].

A histogram over bins [r_l, r_u) with `edges` [B + 1] in nm from 0:

    g_c(b) = V n_c(b) / (P_c (4 pi / 3)(r_u^3 - r_l^3))

where n_c(b) counts the pairs of channel c in bin b, V is the cell volume and P_c is the
channel's number of atom pairs: N_a N_b for two types, N_a (N_a - 1) / 2 for a type with
itself, so an ideal gas reads 1. The channels are the unordered type pairs of `types`
[N] in the order (0, 0), (0, 1), ..., (1, 1), ..., or one channel without types.
`positions` [N, 3] may lie outside the cell. The result has no derivative in the
positions. Syncs with the host, to check the edges and, with `types`, to count them.

Raises `ValueError` if the last edge lies beyond the cutoff of `pairs`.

### observables.temperature

```python
observables.temperature(kinetic: Tensor, degrees_of_freedom: int) -> Tensor
```

Return T = 2K / (g k_B) in K, for K in kJ/mol and g degrees of freedom.

With zero total momentum, g = 3N - 3.

### observables.kinetic_energy

```python
observables.kinetic_energy(velocities: Tensor, masses: Tensor) -> Tensor
```

Return K = sum m v^2 / 2 in kJ/mol for velocities [N, 3] in nm/ps and masses [N] in Da.

### observables.pressure

```python
observables.pressure(kinetic: Tensor, virial: Tensor, cell: Tensor) -> Tensor
```

Return the instantaneous pressure P = (2K + trace W) / (3V) in kJ/mol/nm^3.

`kinetic` K and `virial` W [3, 3] are in kJ/mol; `cell` [3] holds the edge lengths
in nm. Divide by `units.BAR` for bar.

## Periodic geometry

Cells are orthorhombic and periodic. `wrap` maps positions into the cell and reports the lattice
crossings; `displacement` gives minimum-image vectors between positions anywhere.

### space.wrap

```python
space.wrap(positions: Tensor, cell: Tensor) -> tuple[Tensor, Tensor]
```

Wrap positions [..., 3] into the cell, each coordinate in [0, L).

Returns `(wrapped, crossings)`, with crossings [..., 3] int64 such that `positions`
equals `wrapped + crossings * cell` up to rounding.

### space.displacement

```python
space.displacement(left: Tensor, right: Tensor, cell: Tensor) -> Tensor
```

Return the minimum-image vector `left - right` [..., 3] in nm, for positions anywhere.

## Units

Units and physical constants.

NablaDyn works in nm, ps, Da (g/mol), kJ/mol, K and elementary charges e. Multiply a
value by a unit below to convert it into these units; divide to convert back. Numbers and
tensors both work; tensors keep their dtype, device and gradients.

    positions = positions_in_angstrom * units.ANGSTROM   # nm
    pressure_in_bar = pressure / units.BAR
    kT = units.BOLTZMANN * temperature                    # kJ/mol

| Name | Value | Unit and meaning |
| --- | --- | --- |
| `AVOGADRO` | `6.02214076e+23` | 1/mol |
| `BOLTZMANN` | `0.00831446261815324` | kJ/(mol K): k_B N_A, the molar gas constant |
| `COULOMB` | `138.93545764438198` | kJ nm/(mol e^2): N_A e^2 / (4 pi eps0) |
| `ANGSTROM` | `0.1` | nm |
| `FS` | `0.001` | ps |
| `KCAL_PER_MOL` | `4.184` | kJ/mol, the thermochemical calorie |
| `EV` | `96.48533212331002` | kJ/mol: e x 1 V x N_A |
| `BAR` | `0.0602214076` | kJ/(mol nm^3): 1e5 Pa x 1e-27 m^3/nm^3 x N_A |
| `ATM` | `0.0610193412507` | kJ/(mol nm^3) |
| `G_PER_CM3` | `602.214076` | Da/nm^3: 1 g is N_A Da, and 1 cm^3 is 1e21 nm^3 |
