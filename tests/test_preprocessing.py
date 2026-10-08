"""Focused guards for grouping disconnected customer strokes."""
import unittest

import cv2
import numpy as np

import matcher as vm


class CustomerComponentTests(unittest.TestCase):
    def test_closing_does_not_attach_nearby_annotation(self):
        mask = np.zeros((100,160),dtype=np.uint8)
        cv2.line(mask,(20,50),(135,50),255,1)
        cv2.line(mask,(65,35),(65,48),255,1)
        result = vm._prepare_customer_candidate(mask)
        self.assertEqual(cv2.connectedComponents(result,8)[0],3)

    def test_endpoint_join_requires_continuation_in_both_directions(self):
        a = dict(endpoint_coordinates=[(0,0)],endpoint_directions=[(0.,1.)])
        aligned = dict(endpoint_coordinates=[(0,8)],endpoint_directions=[(0.,-1.)])
        sideways = dict(endpoint_coordinates=[(8,0)],endpoint_directions=[(-1.,0.)])
        self.assertEqual(vm._continuation_join(a,aligned)[0],8.)
        self.assertTrue(np.isinf(vm._continuation_join(a,sideways)[0]))
        # The same evidence under rotation is unchanged.
        rotated_a = dict(endpoint_coordinates=[(0,0)],endpoint_directions=[(1.,0.)])
        rotated_b = dict(endpoint_coordinates=[(8,0)],endpoint_directions=[(-1.,0.)])
        self.assertEqual(vm._continuation_join(rotated_a,rotated_b)[0],8.)

    def test_group_traversal_matches_original_growth_rule(self):
        import math
        rng = np.random.default_rng(7)
        descriptors = [dict(label=i,x=int(rng.integers(0,150)),y=int(rng.integers(0,150)),
            width=int(rng.integers(2,30)),height=int(rng.integers(2,30)),
            stroke_width=float(rng.uniform(.5,8))) for i in range(40)]
        excluded = {5,9,20}
        for seed in descriptors:
            group = {seed['label']}
            changed = True
            while changed:
                changed = False
                for d in descriptors:
                    if d['label'] in group or d['label'] in excluded:
                        continue
                    near = any(vm._bbox_gap(d,o)<=max(8.,math.hypot(180,180)*.025,
                               2.5*max(d['stroke_width'],o['stroke_width']))
                               for o in descriptors if o['label'] in group)
                    if near and .25<=d['stroke_width']/max(seed['stroke_width'],.1)<=4.:
                        group.add(d['label'])
                        changed = True
            self.assertEqual(vm._component_group(descriptors,seed,excluded,(180,180)),group)

    def test_component_crop_preserves_full_canvas_graph_and_coordinates(self):
        mask = np.zeros((600,800),dtype=np.uint8)
        cv2.polylines(mask,[np.array([[100,190],[100,60],[320,60],[320,190]])],False,255,5)
        cv2.circle(mask,(650,420),18,255,3)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask,8)
        original = vm.skeleton_to_pixel_graph
        for label in range(1,count):
            full = np.where(labels==label,255,0).astype(np.uint8)
            expected = original(vm.make_skeleton(full))
            from unittest.mock import patch
            with patch.object(vm,"skeleton_to_pixel_graph", wraps=original) as graph_call:
                descriptor = vm._component_descriptor(mask,label,stats,labels)
            actual = original(*graph_call.call_args.args,**graph_call.call_args.kwargs)
            self.assertEqual(set(actual),set(expected))
            self.assertEqual({frozenset(e) for e in actual.edges},
                             {frozenset(e) for e in expected.edges})
            self.assertEqual(descriptor['skeleton_length'],len(expected))
            self.assertLess(graph_call.call_args.args[0].size,mask.size//4)

    def test_interior_annotation_is_not_a_path_continuation(self):
        mask = np.zeros((240, 300), dtype=np.uint8)
        cv2.polylines(mask, [np.array([[30, 200], [30, 30], [250, 30]])], False, 255, 5)
        cv2.putText(mask, "C", (140, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
        retained, report = vm._decompose_customer_foreground(mask)
        self.assertTrue(report["absolute_plausibility"])
        self.assertEqual(report["retained_count"], 1)
        self.assertEqual(np.count_nonzero(retained[90:140, 135:180]), 0)
        self.assertGreater(np.count_nonzero(retained[190:205, 25:36]), 0)

    def test_near_terminal_fragment_and_hook_are_preserved(self):
        mask = np.zeros((240, 300), dtype=np.uint8)
        cv2.polylines(mask, [np.array([[30, 200], [30, 30], [180, 30]])], False, 255, 5)
        cv2.polylines(mask, [np.array([[189, 30], [230, 30], [230, 45]])], False, 255, 5)
        for rotated in (mask, np.rot90(mask).copy()):
            retained, report = vm._decompose_customer_foreground(rotated)
            self.assertEqual(report["retained_count"], 2)
            # Original pixels survive; only the accepted endpoint gap is filled.
            self.assertTrue(np.all(retained[rotated > 0] == 255))
            self.assertEqual(len(report["joined_endpoint_gaps"]), 1)
            signature = vm.build_signature(retained)
            self.assertEqual(signature["raw_topology"]["component_count"], 1)
            self.assertGreater(signature["path_length"], 350)
            self.assertEqual(signature["endpoint_count"], 2)


if __name__ == "__main__":
    unittest.main()
