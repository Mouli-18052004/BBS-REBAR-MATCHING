"""Small invariance/regression checks for deterministic centerline matching."""
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

import matcher as vm


class StructuralGeometryTests(unittest.TestCase):
    def test_line_model_preserves_collinear_return_terminal(self):
        from shape_features import primitives
        points = np.c_[np.r_[np.linspace(0,100,70),np.linspace(100,60,30)],np.zeros(100)]
        for angle in (0., .7, 2.4):
            rotation = np.array([[np.cos(angle),-np.sin(angle)],
                                 [np.sin(angle),np.cos(angle)]])
            transformed = points @ rotation * 3.7 + [42.,-19.]
            for path in (transformed,transformed[::-1]):
                model = primitives(path)
                self.assertEqual(model['line_count'],2)
                self.assertEqual(model['arc_count'],0)
        straight = np.c_[np.linspace(0,100,100),np.zeros(100)]
        self.assertEqual(primitives(straight)['line_count'],1)

    def test_multiloop_fallback_does_not_depend_on_cycle_basis_order(self):
        def square(x,size):
            return ([(0,i) for i in range(x,x+size)] + [(i,x+size-1) for i in range(1,size)]
                    + [(size-1,i) for i in range(x+size-2,x-1,-1)]
                    + [(i,x) for i in range(size-2,0,-1)])
        large,small = square(0,30),square(40,8)
        graph = nx.Graph()
        for cycle in (large,small):
            nx.add_path(graph,cycle+[cycle[0]],weight=1.)
        nx.add_path(graph,[(0,i) for i in range(29,41)],weight=1.)
        nx.add_path(graph,[(-i,0) for i in range(10,-1,-1)],weight=1.)
        for cycles in ([small,large],[large,small]):
            with patch.object(nx,'cycle_basis',return_value=cycles):
                path,_ = vm.extract_topological_path(graph)
            self.assertTrue(set(large).issubset(path))

    def test_cleanup_preserves_relative_terminal_geometry_at_two_sizes(self):
        for size in (90, 300):
            graph = nx.Graph()
            nx.add_path(graph, [(0, x) for x in range(size + 1)], weight=1.)
            # A substantial terminal must survive, while a tiny tick is noise.
            hook = max(1, round(size * .12))
            tick = max(1, round(size * .02))
            nx.add_path(graph, [(y, size // 3) for y in range(hook + 1)], weight=1.)
            nx.add_path(graph, [(y, 2 * size // 3) for y in range(tick + 1)], weight=1.)
            cleaned, _ = vm.clean_skeleton_graph(graph)
            self.assertIn((hook, size // 3), cleaned)
            self.assertNotIn((tick, 2 * size // 3), cleaned)
            self.assertEqual(sum(cleaned.degree(n) == 1 for n in cleaned), 3)

    def test_clean_path_coverage_does_not_hide_dropped_stroke(self):
        import cv2
        mask = np.zeros((160,160),dtype=np.uint8)
        cv2.line(mask,(20,20),(20,130),255,3)
        cv2.line(mask,(90,20),(90,100),255,3)
        signature = vm.build_signature(mask)
        self.assertAlmostEqual(signature['path_coverage'],1.)
        self.assertLess(signature['foreground_coverage'],.7)

    def test_adjacent_three_way_pixels_form_one_crossing(self):
        graph = nx.Graph()
        left = [(0,x) for x in range(-30,1)]
        right = [(0,x) for x in range(1,32)]
        nx.add_path(graph,left+right,weight=1.)
        nx.add_path(graph,[(y,0) for y in range(-30,1)],weight=1.)
        nx.add_path(graph,[(y,1) for y in range(0,31)],weight=1.)
        self.assertEqual(sum(graph.degree(n)==3 for n in graph),2)
        cleaned,_ = vm.clean_skeleton_graph(graph)
        self.assertEqual(sum(cleaned.degree(n)==4 for n in cleaned),1)
        self.assertEqual(sum(cleaned.degree(n)==1 for n in cleaned),4)

    def test_right_angle_is_not_counted_three_times(self):
        points = np.r_[np.c_[np.arange(20),np.zeros(20)],
                       np.c_[np.full(20,19),np.arange(1,21)]]
        for scale in (1., 4.):
            bends = vm.extract_bends(vm.calculate_turn_angles(points * scale))
            np.testing.assert_allclose(bends,[90.],atol=1e-4)
            np.testing.assert_allclose(vm.extract_bends(vm.calculate_turn_angles((points*scale)[::-1])),[-90.],atol=1e-4)

    def test_arc_turn_integral_matches_sweep(self):
        theta = np.linspace(0,np.pi,100)
        points = np.c_[np.cos(theta),np.sin(theta)]*100
        total = np.degrees(vm.calculate_turn_angles(points).sum())
        self.assertAlmostEqual(total,180*94/99,places=3)

    @staticmethod
    def signature(points, closed=False):
        return {"trajectory": vm.normalize_trajectory(points), "closed": closed}

    def test_rigid_invariances_preserve_proportions_and_handedness(self):
        points = np.array([[0., 0.], [1., 0.], [1., 2.], [3., 2.]])
        signature = self.signature(points)
        angle = 0.713
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transformed = (points @ rotation * 7.3 + [40, -20])[::-1]
        self.assertLess(vm.trajectory_distance(signature, self.signature(transformed)), 1e-5)
        self.assertGreater(vm.trajectory_distance(signature, self.signature(points * [2, 1])), 5)
        self.assertGreater(vm.trajectory_distance(signature, self.signature(points * [-1, 1])), 5)

    def test_closed_loop_start_offset(self):
        points = np.array([[0., 0.], [3., 0.], [3., 1.], [1., 2.], [0., 1.]])
        shifted = np.roll(points, 2, axis=0)[::-1]
        a = self.signature(np.vstack([points, points[0]]), True)
        b = self.signature(np.vstack([shifted, shifted[0]]), True)
        self.assertLess(vm.trajectory_distance(a, b), 1e-5)

    def test_loop_stem_path_has_no_jump(self):
        # A >18-edge cycle with an attached stem, independent of catalog IDs.
        cycle = ([(0, x) for x in range(8)] + [(y, 7) for y in range(1, 8)]
                 + [(7, x) for x in range(6, -1, -1)] + [(y, 0) for y in range(6, 0, -1)])
        graph = nx.Graph()
        nx.add_path(graph, cycle + [cycle[0]], weight=1.)
        nx.add_path(graph, [(0, -4), (0, -3), (0, -2), (0, -1), (0, 0)], weight=1.)
        path, loops = vm.extract_topological_path(graph)
        self.assertEqual(loops, 1)
        self.assertTrue(all(graph.has_edge(a, b) for a, b in zip(path, path[1:])))
        self.assertEqual(len(path), graph.number_of_edges() + 1)

    def test_bend_centroids_reverse_consistently(self):
        points = vm.smooth_path(vm.resample_path([(0, 0), (0, 30), (60, 30), (60, 50)], 100))
        turns = vm.calculate_turn_angles(points)
        ratios, positions, bends = vm._ordered_path_features(points, turns)
        self.assertGreater(len(bends), 0)
        reverse_ratios, reverse_positions, reverse_bends = vm._ordered_path_features(points[::-1], vm.calculate_turn_angles(points[::-1]))
        np.testing.assert_allclose(ratios, reverse_ratios[::-1], atol=1e-6)
        np.testing.assert_allclose(positions, [1 - x for x in reverse_positions[::-1]], atol=1e-6)
        np.testing.assert_allclose(bends, [-x for x in reverse_bends[::-1]], atol=1e-4)

    def test_legacy_signature_fallback(self):
        self.assertIsNone(vm.trajectory_distance({"closed": False}, {"closed": False}))

    def test_full_foreground_fingerprint_preserves_extra_structure(self):
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[5:20, 5] = 255
        mask[19, 5:13] = 255
        self.assertEqual(vm.geometry_fingerprint(mask), vm.geometry_fingerprint(np.rot90(np.pad(mask, 8))))
        branched = mask.copy()
        branched[10, 5:11] = 255
        self.assertNotEqual(vm.geometry_fingerprint(mask), vm.geometry_fingerprint(branched))

    def test_equivalent_outside_shortlist_limits_certainty(self):
        signature = dict(endpoint_count=2, junction_count=0, loop_count=0,
                         detected_threads=0, net_rotation=0., total_curvature=0., bends=[])
        catalog = [dict(shape_id=name, signature=signature, threads=0, geometry_fingerprint="same")
                   for name in ("first", "second")]
        with patch.object(vm, "read_customer_image", return_value=np.zeros((1, 1))), \
             patch.object(vm, "prepare_customer", return_value=None), \
             patch.object(vm, "build_signature", return_value=signature), \
             patch.object(vm, "load_or_build_cache", return_value=catalog), \
             patch.object(vm, "compare_stage1", return_value=0.), \
             patch.object(vm, "compare_stage2", return_value=(0., 0., .98)), \
             patch.object(vm, "STAGE1_SHORTLIST", 1):
            stage1, final = vm.match_shape("unused")
        self.assertEqual(len(stage1), 2)
        self.assertEqual(len(final), 1)
        self.assertEqual(final[0]["confidence"], .5)
        self.assertEqual(final[0]["geometry_equivalents"], ["first", "second"])


if __name__ == "__main__":
    unittest.main()
