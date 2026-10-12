"""Angular SchNet: SchNet whose atoms also start from angle features of their neighbors."""

import math
from numbers import Integral

import torch
from torch import Tensor

from ...neighbors import NeighborPairs
from .._schnet._potential import SchNet, _linear, _pair_geometry, _shifted_softplus

# In units of the radial channels squared, whose sums T are of order 1 in a dense liquid: keeps
# the correlations' denominator above 0 for an atom without neighbors.
_FLOOR = 1e-4
_SQRT2 = math.sqrt(2.0)
_SQRT3_2 = math.sqrt(1.5)


def _check_channels(channels: int) -> None:
    """Raise ``ValueError`` unless channels is an integer of at least 1."""
    if isinstance(channels, bool) or not isinstance(channels, Integral) or channels < 1:
        raise ValueError("channels must be an integer of at least 1")


def _check_orders(orders: int) -> None:
    """Raise ``ValueError`` unless orders is an integer of at least 2."""
    if isinstance(orders, bool) or not isinstance(orders, Integral) or orders < 2:
        raise ValueError("orders must be an integer of at least 2")


def _harmonics(unit: Tensor, top: int) -> list[Tensor]:
    """Return the real harmonics C_0 to C_top of unit vectors [P, 3], C_n [P, 2n + 1].

    They are the regular solid harmonics in Racah's normalization at r = 1, from their
    recursion in x, y and z, with the components of m > 0 scaled by sqrt(2), so that
    C_n(u) . C_n(v) = P_n(u . v), the Legendre polynomial, and C_n(-u) = (-1)^n C_n(u).
    """
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    cosine, sine = [[torch.ones_like(x)]], [[torch.zeros_like(x)]]
    for order in range(top):
        c_next, s_next = [], []
        for m in range(order + 1):
            c = (2 * order + 1) * z * cosine[order][m]
            s = (2 * order + 1) * z * sine[order][m]
            if m < order:
                f = math.sqrt((order + m) * (order - m))
                c = c - f * cosine[order - 1][m]
                s = s - f * sine[order - 1][m]
            g = math.sqrt((order + m + 1) * (order - m + 1))
            c_next.append(c / g)
            s_next.append(s / g)
        h = math.sqrt((2 * order + 1) / (2 * order + 2))
        c_next.append(h * (x * cosine[order][order] - y * sine[order][order]))
        s_next.append(h * (y * cosine[order][order] + x * sine[order][order]))
        cosine.append(c_next)
        sine.append(s_next)
    result = []
    for order in range(top + 1):
        parts = [cosine[order][0]]
        for m in range(1, order + 1):
            parts += [_SQRT2 * cosine[order][m], _SQRT2 * sine[order][m]]
        result.append(torch.stack(parts, dim=-1))
    return result


def _traceless(unit: Tensor) -> Tensor:
    """Return u u^T - I/3 of unit vectors [C, 3] as its 5 components [C, 5].

    The components are in an orthonormal basis of the traceless symmetric matrices, so
    Tr(A B) of two such matrices is the dot product of their components.
    """
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]
    return torch.stack(
        (
            (x * x - y * y) / _SQRT2,
            _SQRT3_2 * (z * z - 1.0 / 3.0),
            _SQRT2 * x * y,
            _SQRT2 * x * z,
            _SQRT2 * y * z,
        ),
        dim=-1,
    )


class AngularSchNet(SchNet):
    """SchNet with angle features of each atom's neighbors added to its starting features.

    Built by ``potential.angular_schnet``; see its docstring.
    """

    def __init__(
        self,
        *,
        atom_types: Tensor,
        num_types: int,
        cutoff: float,
        features: int,
        interactions: int,
        gaussians: int,
        channels: int,
        orders: int,
        trainable: bool,
        seed: int,
        dtype: torch.dtype,
    ):
        _check_channels(channels)
        _check_orders(orders)
        super().__init__(
            atom_types=atom_types,
            num_types=num_types,
            cutoff=cutoff,
            features=features,
            interactions=interactions,
            gaussians=gaussians,
            trainable=trainable,
            seed=seed,
            dtype=dtype,
        )
        self.orders = int(orders)
        pairs = channels * (channels + 1) // 2
        count = channels + 2 * pairs + (self.orders - 2) * pairs
        self.angle_radial = _linear(gaussians, channels, bias=False)
        self.angle1 = _linear(count, features)
        self.angle2 = _linear(features, features)
        # A second generator, so the SchNet part is the network schnet draws from the seed.
        generator = torch.Generator().manual_seed(int(seed) + 1)
        with torch.no_grad():
            for layer in (self.angle_radial, self.angle1, self.angle2):
                for name, parameter in layer.named_parameters():
                    if name == "weight":
                        bound = math.sqrt(6.0 / sum(parameter.shape))
                        uniform = torch.rand(
                            parameter.shape,
                            generator=generator,
                            dtype=torch.float64,
                        )
                        values = (2.0 * uniform - 1.0) * bound
                    else:
                        values = torch.zeros_like(parameter)
                    parameter.copy_(values)
                layer.to(device=self.device, dtype=self.dtype)
                layer.requires_grad_(trainable)
        upper = torch.triu_indices(channels, channels, device=self.device)
        self.register_buffer("_upper", upper, persistent=False)
        # Where (c, c) sits among the c <= d pairs, in the same row-major order.
        diagonal = [c * channels - c * (c - 1) // 2 for c in range(channels)]
        self.register_buffer(
            "_diagonal",
            torch.tensor(diagonal, device=self.device),
            persistent=False,
        )

    def _angle_features(
        self,
        first: Tensor,
        second: Tensor,
        vectors: Tensor,
        distance: Tensor,
        envelope: Tensor,
        basis: Tensor,
    ) -> Tensor:
        """Return each atom's angle features [N, M + L M(M + 1)/2] from its pairs.

        M channels and L ``orders``: M + M(M + 1) for the orders up to 2, then M(M + 1)/2 for
        each order from 3 to L.
        """
        rows, columns = self._upper[0], self._upper[1]
        # A masked pair has r = 1 and an envelope of 0, so its radial weight is 0 and its
        # direction, finite, adds nothing.
        unit = vectors / distance[:, None]
        radial = self.angle_radial(basis) * envelope[:, None]
        m, k = radial.shape[1], rows.numel()
        higher = _harmonics(unit, self.orders)[3:] if self.orders > 2 else []
        widths = [m * harmonic.shape[1] for harmonic in higher]
        # Per pair: R, R u, R (u u^T - I/3) in 5 components, R_c R_d for c <= d, the envelope,
        # and R C_n for each order n above 2; one scatter sums them over each atom's pairs, the
        # second atom seeing -u, which flips the odd orders.
        per_pair = torch.cat(
            (
                radial,
                (radial[:, :, None] * unit[:, None, :]).reshape(-1, 3 * m),
                (radial[:, :, None] * _traceless(unit)[:, None, :]).reshape(-1, 5 * m),
                radial[:, rows] * radial[:, columns],
                envelope[:, None],
                *(
                    (radial[:, :, None] * harmonic[:, None, :]).reshape(-1, width)
                    for harmonic, width in zip(higher, widths)
                ),
            ),
            dim=1,
        )
        sign = per_pair.new_ones(per_pair.shape[1])
        sign[m : 4 * m] = -1.0
        start = 9 * m + k + 1
        for order, width in enumerate(widths, start=3):
            if order % 2:
                sign[start : start + width] = -1.0
            start += width
        # A fit's vmap over frames maps scatter_add as one sum, index_add one frame at a time;
        # index_put's batched form compiles wrongly for SchNet's messages. Out of place, since
        # the in-place form fails under vmap.
        total = per_pair.shape[1]
        summed = (
            per_pair.new_zeros((self.num_atoms, total))
            .scatter_add(0, first[:, None].expand(-1, total), per_pair)
            .scatter_add(0, second[:, None].expand(-1, total), per_pair * sign)
        )
        s, v, q, t, count, *moments = torch.split(summed, (m, 3 * m, 5 * m, k, 1, *widths), dim=1)
        v, q = v.reshape(-1, m, 3), q.reshape(-1, m, 5)
        # The self-terms j = k, which hold no angle, come out; dividing by the channels' own
        # sums leaves dimensionless three-body correlations, near 0 where directions are random.
        # For order n the self-terms are T, since C_n(u) . C_n(u) = P_n(1) = 1.
        diagonal = t[:, self._diagonal] + _FLOOR
        norm = (diagonal[:, rows] * diagonal[:, columns]).sqrt()
        correlations = [((v[:, rows] * v[:, columns]).sum(-1) - t) / norm]
        correlations.append(((q[:, rows] * q[:, columns]).sum(-1) - 2.0 / 3.0 * t) / norm)
        for moment in moments:
            a = moment.reshape(self.num_atoms, m, -1)
            correlations.append(((a[:, rows] * a[:, columns]).sum(-1) - t) / norm)
        return torch.cat((s / (1.0 + count), *correlations), dim=-1)

    def _energy_wrapped(
        self,
        positions: Tensor,
        cell: Tensor,
        pairs: NeighborPairs,
        strain: Tensor | None = None,
    ) -> Tensor:
        first, second, vectors, distance, envelope, basis = _pair_geometry(
            positions,
            cell,
            pairs,
            strain,
            self._cutoff,
            self._centres,
        )
        features = self._angle_features(first, second, vectors, distance, envelope, basis)
        x = self.embedding[self.atom_types] + self.angle2(
            _shifted_softplus(self.angle1(features)),
        )
        for block in self.blocks:
            x = block(x, basis, envelope, first, second)
        return self.readout2(_shifted_softplus(self.readout1(x))).sum()


__all__ = ("AngularSchNet",)
