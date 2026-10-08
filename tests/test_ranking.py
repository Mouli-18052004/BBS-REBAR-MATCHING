"""Final selection must not contradict identical extracted visual evidence."""
import unittest
import numpy as np
import matcher as vm
from shape_features import apply_confidence, match_decision, primitive_distance


class FinalSelectionTests(unittest.TestCase):
    def test_short_terminal_bend_cannot_be_hidden_by_mean_heading(self):
        from unittest.mock import patch
        base = dict(heading=np.zeros(10),heading_reverse=np.zeros(10),net_rotation=0.,
                    total_curvature=0.,loop_count=0,endpoint_count=2,junction_count=0)
        query = dict(base,bends=[90.,90.,135.])
        correct = dict(base,bends=[89.,91.,134.])
        wrong = dict(base,bends=[90.,90.,90.])
        with patch.object(vm,'trajectory_distance',return_value=.01):
            good = vm.stage2_distance_components(query,correct)
            bad = vm.stage2_distance_components(query,wrong)
        self.assertLess(good['ordered_bends'],1.)
        self.assertAlmostEqual(bad['ordered_bends'],36.)

    def test_ordered_bends_reverse_phase_and_missing_trace(self):
        q = dict(bends=[90.,-45.,135.])
        self.assertEqual(vm.ordered_bend_disagreement(q,dict(bends=[-135.,45.,-90.])),0.)
        self.assertEqual(vm.ordered_bend_disagreement(dict(q,closed=True),
                         dict(bends=[135.,90.,-45.],closed=True)),0.)
        self.assertGreater(vm.ordered_bend_disagreement(q,dict(bends=[90.,135.,-45.])),0.)
        self.assertEqual(vm.ordered_bend_disagreement(dict(q,path_coverage=.5),dict(bends=[10.])),0.)

    def test_arc_partition_uncertainty_remains_bounded(self):
        a = dict(primitives=dict(line_count=0,sequence=[dict(kind='arc',start=0.,end=1.,sweep=125.)]))
        b = dict(primitives=dict(line_count=1,sequence=[dict(kind='line',start=0.,end=.18,sweep=0.),
                                                     dict(kind='arc',start=.18,end=1.,sweep=108.)]))
        self.assertAlmostEqual(vm._arc_evidence_distance(a,b,.3),.3)

    def test_global_alignment_cannot_cap_arc_mismatch(self):
        from unittest.mock import patch
        base = dict(heading=np.zeros(10),heading_reverse=np.zeros(10),net_rotation=0.,
                    total_curvature=0.,loop_count=0,endpoint_count=2,junction_count=0)
        arc = dict(base,primitives=dict(line_count=0,sequence=[dict(kind='arc',start=0.,end=1.,sweep=90.)]))
        line = dict(base,primitives=dict(line_count=1,sequence=[dict(kind='line',start=0.,end=1.,sweep=0.)]))
        with patch.object(vm,'trajectory_distance',return_value=.01):
            terms = vm.stage2_distance_components(arc,line)
        self.assertEqual(terms['arc_geometry'],20.)
        self.assertAlmostEqual(terms['line_count'],.01)

    def test_dropped_foreground_requires_review_even_with_perfect_score(self):
        rows = [dict(structural_dist=0.,confidence=.98)]
        self.assertEqual(match_decision(rows,{'foreground_coverage':.6,'path_coverage':1.})[0],'review')

    def test_invalid_scores_cannot_be_accepted(self):
        for value in (float('nan'),float('inf'),-1.):
            rows = [dict(structural_dist=value,confidence=.98)]
            self.assertEqual(match_decision(rows,{})[0],'review')
            with self.assertRaises(ValueError):
                apply_confidence(rows)

    def test_no_match_boundary_does_not_confuse_ambiguity_with_absence(self):
        self.assertEqual(match_decision([dict(structural_dist=90., confidence=0.)],{})[0],'no_match')
        self.assertEqual(match_decision([dict(structural_dist=89.99, confidence=0.)],{})[0],'review')
        self.assertEqual(match_decision([dict(structural_dist=0., confidence=0.)],{})[0],'review')
        self.assertEqual(match_decision([dict(structural_dist=100., confidence=0.)],{'path_coverage':.5})[0],'review')

    def test_arc_sweep_distinguishes_angles_with_same_count(self):
        def arc(sweep):
            return {'primitives': {'line_count':0, 'sequence':[
                {'kind':'arc','start':0.,'end':1.,'sweep':sweep}]}}
        self.assertEqual(primitive_distance(arc(90),arc(-90)),0.)
        self.assertGreater(primitive_distance(arc(90),arc(180)),0.)

    def test_nearly_exact_match_does_not_need_large_absolute_gap(self):
        rows = [dict(structural_dist=.01), dict(structural_dist=1.)]
        apply_confidence(rows)
        self.assertGreater(rows[0]['confidence'], .95)
        self.assertEqual(match_decision(rows, {})[0], 'matched')

    def test_exact_tie_is_still_ambiguous(self):
        rows = [dict(structural_dist=0.), dict(structural_dist=0.)]
        apply_confidence(rows)
        self.assertEqual(match_decision(rows, {})[0], 'review')

    def test_close_runner_up_cannot_display_higher_confidence(self):
        rows = [dict(structural_dist=d) for d in (5., 5.1, 20., 60.)]
        apply_confidence(rows)
        self.assertEqual([r['confidence'] for r in rows], sorted(
            [r['confidence'] for r in rows], reverse=True))
        self.assertGreater(rows[0]['fit_score'], rows[1]['fit_score'])
        self.assertEqual(match_decision(rows, {})[0], 'review')

    def test_clear_good_match_and_complete_bad_match_have_distinct_states(self):
        for distances, expected in (((2., 30.), 'matched'), ((95., 120.), 'no_match')):
            rows = [dict(structural_dist=d) for d in distances]
            apply_confidence(rows)
            self.assertEqual(match_decision(rows, {})[0], expected)
        self.assertEqual(match_decision(rows, {'path_coverage': .5})[0], 'review')

    def test_equivalent_geometry_caps_all_candidates_consistently(self):
        rows = [dict(structural_dist=0., geometry_equivalents=['a','b']), dict(structural_dist=25.)]
        apply_confidence(rows)
        self.assertEqual(rows[0]['confidence'], .5)
        self.assertLessEqual(rows[1]['confidence'], .5)
        self.assertEqual(match_decision(rows, {})[0], 'review')

    @staticmethod
    def signature(threads=0):
        path = np.c_[np.linspace(0,1,100),np.zeros(100)]
        return dict(endpoint_count=2,junction_count=0,loop_count=0,closed=False,
            detected_threads=threads,net_rotation=0.,total_curvature=0.,bends=[],
            heading=np.zeros(94),heading_reverse=np.zeros(94),
            trajectory=vm.normalize_trajectory(path))

    def test_identical_visuals_have_zero_distance_despite_metadata_conflict(self):
        signature = self.signature()
        for metadata_threads in (0,1,2,None):
            terms = vm.stage2_distance_components(signature,signature,metadata_threads)
            self.assertEqual(terms['thread'],0.)
            self.assertAlmostEqual(vm.compare_stage2(signature,signature,metadata_threads)[1],0.,places=5)

    def test_real_visual_thread_difference_remains_discriminative(self):
        query = self.signature(0)
        candidate = self.signature(1)
        self.assertEqual(vm.stage2_distance_components(query,candidate,0)['thread'],30.)
        self.assertEqual(vm.stage2_distance_components(candidate,query,2)['thread'],30.)

    def test_legacy_signature_can_fall_back_to_metadata(self):
        query = self.signature()
        candidate = self.signature()
        del candidate['detected_threads']
        self.assertEqual(vm.stage2_distance_components(query,candidate,1)['thread'],30.)


if __name__=='__main__':
    unittest.main()
