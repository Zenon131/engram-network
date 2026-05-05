import torch
import unittest
from dragon.layers import LinearAttention, BDHLayer, SpatialBDHLayer
from dragon.spatial import NeuronSpatialMap


class TestLinearAttention(unittest.TestCase):
    def setUp(self):
        self.n_neurons = 64
        self.d_model = 32
        self.layer = LinearAttention(self.n_neurons, self.d_model)

    def test_forward_shape(self):
        batch_size = 4
        x = torch.randn(batch_size, self.n_neurons)
        y = torch.randn(batch_size, self.n_neurons)
        rho = torch.randn(batch_size, self.n_neurons, self.d_model)

        new_y, new_rho = self.layer(x, y, rho)

        self.assertEqual(new_y.shape, (batch_size, self.n_neurons))
        self.assertEqual(new_rho.shape, (batch_size, self.n_neurons, self.d_model))


class TestBDHLayer(unittest.TestCase):
    def setUp(self):
        self.n_neurons = 64
        self.d_model = 32
        self.layer = BDHLayer(self.n_neurons, self.d_model)

    def test_forward_shape(self):
        batch_size = 4
        x = torch.randn(batch_size, self.n_neurons)
        y = torch.randn(batch_size, self.n_neurons)
        rho = torch.randn(batch_size, self.n_neurons, self.d_model)

        new_x, new_y, new_rho = self.layer(x, y, rho)

        self.assertEqual(new_x.shape, (batch_size, self.n_neurons))
        self.assertEqual(new_y.shape, (batch_size, self.n_neurons))
        self.assertEqual(new_rho.shape, (batch_size, self.n_neurons, self.d_model))


class TestNeuronSpatialMap(unittest.TestCase):
    """Tests for the distance-based neuron clustering module."""

    def setUp(self):
        self.n_neurons = 64
        self.n_clusters = 8
        self.spatial_dim = 2
        self.spatial_map = NeuronSpatialMap(
            n_neurons=self.n_neurons,
            n_clusters=self.n_clusters,
            spatial_dim=self.spatial_dim,
            sigma=1.0,
            decay_type="gaussian",
        )

    def test_assignments_cover_all_neurons(self):
        """Every neuron should be assigned to exactly one cluster."""
        self.assertEqual(self.spatial_map.assignments.shape, (self.n_neurons,))
        self.assertTrue((self.spatial_map.assignments >= 0).all())
        self.assertTrue((self.spatial_map.assignments < self.n_clusters).all())

    def test_cluster_sizes_sum_to_n(self):
        self.assertEqual(int(self.spatial_map.cluster_sizes.sum().item()), self.n_neurons)

    def test_positions_shape(self):
        self.assertEqual(
            self.spatial_map.positions.shape, (self.n_clusters, self.spatial_dim)
        )

    def test_decay_matrix_shape_and_range(self):
        decay = self.spatial_map.compute_decay_matrix()
        self.assertEqual(decay.shape, (self.n_clusters, self.n_clusters))
        # Diagonal should be ~1 (self-distance = 0)
        for i in range(self.n_clusters):
            self.assertAlmostEqual(decay[i, i].item(), 1.0, places=3)
        # All values >= min_decay
        self.assertTrue((decay >= self.spatial_map.min_decay - 1e-6).all())
        # All values <= 1
        self.assertTrue((decay <= 1.0 + 1e-6).all())

    def test_decay_symmetry(self):
        decay = self.spatial_map.compute_decay_matrix()
        self.assertTrue(torch.allclose(decay, decay.T, atol=1e-6))

    def test_interaction_scale_shape(self):
        B = 4
        source = torch.randn(B, self.n_neurons)
        scale = self.spatial_map.get_interaction_scale(source)
        self.assertEqual(scale.shape, (B, self.n_neurons))

    def test_interaction_scale_mean_near_one(self):
        """Scales should be normalised so mean ≈ 1."""
        B = 4
        source = torch.randn(B, self.n_neurons).abs()
        scale = self.spatial_map.get_interaction_scale(source)
        mean_scale = scale.mean(dim=1)
        for b in range(B):
            self.assertAlmostEqual(mean_scale[b].item(), 1.0, places=1)

    def test_modulate_preserves_shape(self):
        B = 4
        source = torch.randn(B, self.n_neurons)
        target = torch.randn(B, self.n_neurons)
        result = self.spatial_map.modulate(source, target)
        self.assertEqual(result.shape, target.shape)

    def test_nearby_clusters_stronger_than_distant(self):
        """Neurons in the same cluster should get higher scale than distant ones."""
        # Create a source that's concentrated in cluster 0
        source = torch.zeros(1, self.n_neurons)
        cluster_0_mask = self.spatial_map.assignments == 0
        source[0, cluster_0_mask] = 10.0

        scale = self.spatial_map.get_interaction_scale(source)

        # Mean scale for cluster 0 neurons should be higher than distant clusters
        scale_c0 = scale[0, cluster_0_mask].mean()
        # Pick the furthest cluster
        furthest_cluster = self.n_clusters - 1
        cluster_far_mask = self.spatial_map.assignments == furthest_cluster
        scale_cfar = scale[0, cluster_far_mask].mean()
        self.assertGreater(scale_c0.item(), scale_cfar.item())

    def test_different_decay_types(self):
        for dtype in ["gaussian", "exponential", "inverse", "cosine"]:
            smap = NeuronSpatialMap(
                n_neurons=64, n_clusters=8, spatial_dim=2, decay_type=dtype,
            )
            decay = smap.compute_decay_matrix()
            self.assertEqual(decay.shape, (8, 8))
            self.assertTrue((decay >= smap.min_decay - 1e-6).all())

    def test_3d_positions(self):
        smap = NeuronSpatialMap(
            n_neurons=64, n_clusters=8, spatial_dim=3,
        )
        self.assertEqual(smap.positions.shape, (8, 3))

    def test_cluster_stats(self):
        stats = self.spatial_map.get_cluster_stats()
        self.assertIn("sigma", stats)
        self.assertIn("mean_decay", stats)
        self.assertIn("n_clusters", stats)
        self.assertGreater(stats["sigma"], 0)


class TestSpatialBDHLayer(unittest.TestCase):
    """Tests for the BDH layer with spatial distance modulation."""

    def setUp(self):
        self.n_neurons = 64
        self.d_model = 32
        self.n_clusters = 8
        self.spatial_map = NeuronSpatialMap(
            n_neurons=self.n_neurons,
            n_clusters=self.n_clusters,
            spatial_dim=2,
        )
        self.layer = SpatialBDHLayer(
            self.n_neurons, self.d_model, self.spatial_map,
        )

    def test_forward_shape(self):
        B = 4
        x = torch.randn(B, self.n_neurons)
        y = torch.randn(B, self.n_neurons)
        rho = torch.randn(B, self.n_neurons, self.d_model)

        new_x, new_y, new_rho = self.layer(x, y, rho)

        self.assertEqual(new_x.shape, (B, self.n_neurons))
        self.assertEqual(new_y.shape, (B, self.n_neurons))
        self.assertEqual(new_rho.shape, (B, self.n_neurons, self.d_model))

    def test_gradients_flow(self):
        """Ensure gradients flow through spatial positions and sigma."""
        B = 2
        x = torch.randn(B, self.n_neurons)
        y = torch.randn(B, self.n_neurons)
        rho = torch.randn(B, self.n_neurons, self.d_model)

        new_x, new_y, new_rho = self.layer(x, y, rho)
        loss = new_y.sum()
        loss.backward()

        # Spatial map parameters should have gradients
        self.assertIsNotNone(self.spatial_map.positions.grad)
        self.assertIsNotNone(self.spatial_map.log_sigma.grad)

    def test_selective_modulation(self):
        """Test with only feedforward modulation enabled."""
        layer = SpatialBDHLayer(
            self.n_neurons, self.d_model, self.spatial_map,
            modulate_feedforward=True,
            modulate_hebbian=False,
            modulate_output=False,
        )
        B = 2
        x = torch.randn(B, self.n_neurons)
        y = torch.randn(B, self.n_neurons)
        rho = torch.randn(B, self.n_neurons, self.d_model)

        new_x, new_y, new_rho = layer(x, y, rho)
        self.assertEqual(new_x.shape, (B, self.n_neurons))

    def test_shared_spatial_map_across_layers(self):
        """Multiple layers should share the same spatial map object."""
        layer2 = SpatialBDHLayer(
            self.n_neurons, self.d_model, self.spatial_map,
        )
        self.assertIs(self.layer.spatial_map, layer2.spatial_map)


if __name__ == '__main__':
    unittest.main()