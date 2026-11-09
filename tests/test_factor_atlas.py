import numpy as np
import pytest

from sparsifier.prune.factor_atlas import FactorAtlas, geometry_coupling, geometric_error, orth


def circle_atlas():
    rng = np.random.default_rng(12)
    angles = rng.uniform(-np.pi, np.pi, (240, 2))
    # Known orthogonal factors isolate chart estimation from dictionary recovery.
    y = np.column_stack((np.cos(angles[:, 0]), np.sin(angles[:, 0]),
                         np.cos(angles[:, 1]), np.sin(angles[:, 1])))
    atlas = FactorAtlas(groups=2, width=2, active=2, charts=4, rank=1, seed=3)
    atlas.dictionary.dictionary = np.eye(4)
    atlas.dictionary.mean = np.zeros((4, 1))
    atlas.fit_charts(y)
    return atlas, y


def test_multiple_factors_have_independent_local_assignments():
    atlas, y = circle_atlas()
    frames, active, assignments = atlas.frames(y)
    assert active.all()
    assert len(set(map(tuple, assignments))) > 4
    assert all(q.shape == (4, 2) for q in frames)
    for q in frames:
        np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-8)
    # A sample's assignment cannot depend on other evaluation samples.
    _, _, alone = atlas.assign(y[:1])
    np.testing.assert_array_equal(alone[0], assignments[0])


def test_assignment_shuffle_preserves_occupancy_but_changes_metric():
    atlas, y = circle_atlas()
    frames, active, assignments = atlas.frames(y)
    shuffled, active2, permuted = atlas.frames(y, mode="shuffled", seed=40)
    np.testing.assert_array_equal(active, active2)
    for i in range(2):
        np.testing.assert_array_equal(np.sort(assignments[:, i]), np.sort(permuted[:, i]))
    assert np.any(assignments != permuted)
    x = y + np.array([2, 3, 4, 5])
    assert not np.allclose(geometry_coupling(x, frames), geometry_coupling(x, shuffled))


def test_local_charts_preserve_circle_tangents_better_than_radial_noise():
    atlas, y = circle_atlas()
    frames, _, _ = atlas.frames(y)
    tangent = np.column_stack((-y[:, 1], y[:, 0], -y[:, 3], y[:, 2]))
    assert geometric_error(tangent, frames) > 1.5
    assert geometric_error(y, frames) < .5
    assert all(q.shape[1] == 4 for q in atlas.frames(y, mode="factor")[0])
    assert all(q.shape[1] == 2 for q in atlas.frames(y, mode="rank_matched")[0])


def test_rank_and_alignment_fail_closed():
    assert orth(np.zeros((4, 3))).shape == (4, 0)
    with pytest.raises(ValueError):
        geometry_coupling(np.ones((3, 2)), [np.eye(2)])


def test_atlas_roundtrip_preserves_assignments_and_frames(tmp_path):
    atlas, y = circle_atlas()
    path = tmp_path / "atlas.npz"
    atlas.save(path)
    restored = FactorAtlas.load(path)
    frames, active, assignments = atlas.frames(y[:10])
    other, a2, c2 = restored.frames(y[:10])
    np.testing.assert_array_equal(active, a2)
    np.testing.assert_array_equal(assignments, c2)
    for q, r in zip(frames, other):
        np.testing.assert_allclose(q @ q.T, r @ r.T)


def test_dead_factor_is_reported_and_never_gets_fake_tangents():
    atlas, y = circle_atlas()
    atlas.dictionary.dictionary[:, 2:] = 0
    atlas.fit_charts(y)
    assert not atlas.factors[1].bases
    assert 1 in atlas.diagnostics(y)["unsupported_factors"]
    frames, _, assignments = atlas.frames(y)
    assert (assignments[:, 1] == -1).all()
    assert all(q.shape[1] <= 1 for q in frames)
