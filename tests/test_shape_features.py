import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import networkx as nx
import numpy as np
import matcher as vm
from shape_features import primitives, primitive_distance, parameter_distance, match_decision


class PrimitiveTests(unittest.TestCase):
    def feature(self, path):
        return primitives(vm.resample_path(path,100))

    def test_sharp_bend_is_not_circular_arc(self):
        p = self.feature([(0,0),(0,100),(100,100)])
        self.assertEqual((p['line_count'],p['arc_count'],p['bend_count']),(2,0,1))

    def test_small_squared_return_is_not_semicircle_in_large_drawing(self):
        p = self.feature([(20,0),(0,0),(0,20),(150,20),(150,150)])
        self.assertEqual(p['arc_count'],0)

    def test_circle_and_semicircle(self):
        for angle in (np.pi,2*np.pi):
            t = np.linspace(0,angle,100)
            p = primitives(np.c_[np.cos(t),np.sin(t)])
            self.assertEqual(p['arc_count'],1)
            self.assertEqual(p['line_count'],0)
            self.assertAlmostEqual(abs(p['sequence'][0]['sweep']),np.degrees(angle),places=3)

    def test_invariance_and_reversal(self):
        path = np.array([(0,0),(0,100),(80,100),(120,150)],float)
        a = self.feature(path)
        t = .713
        rotate = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        b = self.feature((path @ rotate * 4.2 + [5,10])[::-1])
        self.assertEqual((a['line_count'],a['arc_count']),(b['line_count'],b['arc_count']))
        np.testing.assert_allclose([s['ratio'] for s in a['sequence']],
                                   [s['ratio'] for s in b['sequence'][::-1]],atol=.04)

    def test_slight_wobble_does_not_make_many_lines(self):
        x = np.linspace(0,100,100)
        p = primitives(np.c_[x,.2*np.sin(x)])
        self.assertEqual(p['line_count'],1)
        self.assertEqual(p['arc_count'],0)

    def test_crossing_keeps_both_terminals_and_all_edges(self):
        # Open bar crosses itself at the origin, with a loop in between.
        g = nx.Graph()
        path = [(0,x) for x in range(-30,31)]
        path += [(y,30) for y in range(1,31)]
        path += [(30,x) for x in range(29,-1,-1)]
        path += [(y,0) for y in range(29,-31,-1)]
        nx.add_path(g,path,weight=1.)
        traced, loops = vm.extract_topological_path(g)
        self.assertEqual(loops,1)
        self.assertEqual(len(traced),g.number_of_edges()+1)
        self.assertEqual({traced[0],traced[-1]},{(0,-30),(-30,0)})
        self.assertTrue(all(g.has_edge(a,b) for a,b in zip(traced,traced[1:])))

    def test_missing_annotations_are_neutral(self):
        self.assertEqual(parameter_distance(None,'A B'),0)
        self.assertEqual(parameter_distance('A B',None),0)
        self.assertEqual(parameter_distance('A,B','B A'),0)
        self.assertGreater(parameter_distance('C','c'),0)
        self.assertGreater(parameter_distance('A B','A B C'),0)

    def test_line_count_participates_in_distance(self):
        a = {'primitives':{'sequence':[],'line_count':2}}
        b = {'primitives':{'sequence':[],'line_count':4}}
        self.assertGreater(primitive_distance(a,b),0)

    def test_lower_page_score_can_preserve_more_complete_stroke(self):
        short = np.eye(10,dtype=np.uint8)
        full = np.eye(20,dtype=np.uint8)
        def decompose(mask):
            return mask, {'absolute_plausibility':True,'primary_component_metrics':{'skeleton_length':len(mask)}}
        with patch.object(vm,'_customer_foreground_candidates',return_value={'short':short,'full':full}), \
             patch.object(vm,'_prepare_customer_candidate',side_effect=lambda x:x), \
             patch.object(vm,'_customer_candidate_metrics',side_effect=lambda mask: {'score':8 if len(mask)==10 else 3,'skeleton_pixels':20}), \
             patch.object(vm,'_decompose_customer_foreground',side_effect=decompose), \
             patch.object(vm,'canonicalize_geometry',side_effect=lambda x:x):
            self.assertIs(vm.prepare_customer(np.zeros((20,20))),full)

    def test_reject_and_review_are_not_confirmations_of_novelty(self):
        status,message = match_decision([dict(structural_dist=120,confidence=.9)],{})
        self.assertEqual(status,'no_match')
        self.assertIn('may be',message)
        self.assertEqual(match_decision([dict(structural_dist=5,confidence=.9)],{'path_coverage':.5})[0],'review')
        self.assertEqual(match_decision([dict(structural_dist=5,confidence=.1)],{})[0],'review')

    def test_verified_annotation_overlay_preserves_original(self):
        items = [dict(shape_id='sample',parameters='')]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'data').mkdir()
            (root/'data/catalog_parameters.json').write_text(json.dumps({'sample':{'labels':['A','B'],'source':'test fixture'}}))
            with patch.object(vm,'PROJECT_ROOT',root):
                enriched = vm._with_catalog_annotations(items)
            self.assertEqual(items[0]['parameters'],'')
            self.assertEqual(enriched[0]['parameters'],['A','B'])


if __name__ == '__main__':
    unittest.main()
