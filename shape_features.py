"""Deterministic line/circular-arc models and optional trusted annotations."""
import math
import re
import numpy as np
import cv2

# One shared operating boundary; this is not a probability of catalog absence.
NO_MATCH_DISTANCE = 90.0
MIN_GEOMETRIC_FIT = .98 * math.exp(-NO_MATCH_DISTANCE / 30.)


def primitives(points):
    """Minimum-description piecewise line/arc fit at fixed spatial tolerance.

    Endpoints and order are retained. Small stroke deviations within 1% of
    the bounding diagonal are drawing uncertainty, not extra line segments.
    Arc fits must cover >=20 degrees with monotonic turning and beat a line.
    Counts are estimates at this resolution, not catalog metadata truth.
    """
    p = np.asarray(points, dtype=float)
    if len(p) < 6:
        return {'sequence': [], 'line_count': 0, 'arc_count': 0, 'bend_count': 0}
    scale = max(float(np.linalg.norm(np.ptp(p, axis=0))), 1e-9)
    tolerance = .01 * scale
    # Knot grid limits complexity while retaining the original endpoints.
    vertices = cv2.approxPolyDP(p.astype(np.float32), tolerance, False).reshape(-1,2)
    corners = {int(np.argmin(np.linalg.norm(p-v,axis=1))) for v in vertices}
    knots = sorted(set(range(0, len(p), 3)) | corners | {len(p)-1})
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    total = max(float(distance[-1]), 1e-9)
    best = [float('inf')] * len(knots)
    paths = [[] for _ in knots]
    best[0] = 0.
    for j in range(1, len(knots)):
        for i in range(j):
            start, end = knots[i], knots[j]
            q = p[start:end+1]
            chord = q[-1]-q[0]
            span = np.linalg.norm(chord)
            if span < 1e-9:
                line_error = float('inf')
            else:
                # Fit a finite, traversed segment rather than an infinite line.
                # An out-and-back hook is collinear but is not one segment.
                projection = (q-q[0]) @ chord / span**2
                closest = q[0] + np.clip(projection, 0., 1.)[:,None]*chord
                line_error = float(np.mean(np.sum((q-closest)**2, axis=1)))
                backtrack = float(np.maximum(-np.diff(projection)*span, 0.).sum())
                if backtrack > tolerance:
                    line_error = float('inf')
            choices = [('line', line_error, 2., 0.)]
            if len(q) >= 10:
                center_q = q-q.mean(axis=0)
                a = np.c_[2*center_q, np.ones(len(q))]
                coef, _, _, _ = np.linalg.lstsq(a, np.sum(center_q**2, axis=1), rcond=None)
                radius2 = coef[2]+np.sum(coef[:2]**2)
                if radius2 > 0:
                    radial = center_q-coef[:2]
                    angles = np.unwrap(np.arctan2(radial[:,1], radial[:,0]))
                    sweep = float(angles[-1]-angles[0])
                    monotonic = abs(sweep) / max(float(np.abs(np.diff(angles)).sum()), 1e-9)
                    arc_error = float(np.mean((np.linalg.norm(radial,axis=1)-math.sqrt(radius2))**2))
                    # Straight spans joined at a corner cannot form a good
                    # circle fit throughout; radial error rejects that model.
                    # Judge circularity relative to this primitive, not the
                    # whole drawing. Otherwise a small squared U can pass as
                    # a semicircle simply because the overall bar is large.
                    local_tolerance = .01 * max(float(np.linalg.norm(np.ptp(q,axis=0))),1e-9)
                    if (abs(sweep) >= math.radians(20) and monotonic > .9
                            and arc_error < line_error and math.sqrt(arc_error) <= local_tolerance):
                        choices.append(('arc', arc_error, 3., math.degrees(sweep)))
            for kind, error, parameters, sweep in choices:
                if math.sqrt(error) > tolerance:
                    continue
                cost = best[i] + parameters * math.log(len(p)) + len(q)*error/tolerance**2
                if cost < best[j]:
                    best[j] = cost
                    paths[j] = paths[i] + [dict(kind=kind, start=float(distance[start]/total),
                        end=float(distance[end]/total), ratio=float((distance[end]-distance[start])/total),
                        sweep=sweep)]
    sequence = paths[-1]
    return {'sequence': sequence, 'line_count': sum(s['kind']=='line' for s in sequence),
            'arc_count': sum(s['kind']=='arc' for s in sequence),
            'bend_count': sum(a['kind']==b['kind']=='line' for a,b in zip(sequence,sequence[1:]))}


def primitive_distance_components(a, b):
    """Arc occupancy mismatch at fixed arc length, in angular score units.

    20 degrees is the existing minimum resolved bend, not an ID penalty.
    Counts are reported, while this continuous term avoids brittle integer
    count penalties at split/merge boundaries. Closed paths allow phase shifts.
    """
    if not a.get('primitives') or not b.get('primitives'):
        return {'arc_geometry': 0., 'line_count': 0.}
    if min(a.get('path_coverage',1.),b.get('path_coverage',1.)) < .95:
        return {'arc_geometry': 0., 'line_count': 0.}
    positions = np.linspace(0,1,100)
    def profile(sig):
        result = np.zeros(100)
        sweeps = np.zeros(100)
        for segment in sig['primitives']['sequence']:
            if segment['kind']=='arc':
                active = (positions>=segment['start']) & (positions<=segment['end'])
                result[active] = 1.
                sweeps[active] = segment.get('sweep',0.)
        return result,sweeps
    (x, sx),(y, sy) = profile(a),profile(b)
    shifts = range(100) if a.get('closed') and b.get('closed') else (0,)
    # An arc's location alone does not describe its geometry: retain signed
    # sweep too. Traversal reversal reverses order AND turn sign. Rotation
    # leaves sweep unchanged. Compare both features under the same traversal.
    occupancy = min(float(np.mean(np.abs(x-np.roll(z,k)) +
        np.minimum(np.abs(sx-np.roll(sz,k))/180.,1.) * x * np.roll(z,k)))
        for z,sz in ((y,sy),(y[::-1],-sy[::-1])) for k in shifts)
    na,nb = a['primitives']['line_count'],b['primitives']['line_count']
    return {'arc_geometry': 20.*occupancy,
            'line_count': 20.*abs(na-nb)/max(na,nb,1)}


def primitive_distance(a, b):
    return sum(primitive_distance_components(a, b).values())


def parameter_labels(value):
    if not value:
        return set()
    if isinstance(value, (list, tuple, set)):
        value = ' '.join(map(str,value))
    # Case can be meaningful: C=arc length and c=radius in engineering notes.
    return set(re.findall(r'(?<![A-Za-z])[A-Za-z](?![A-Za-z])', str(value)))


def parameter_distance(query, candidate):
    """Missing annotations are neutral. Only explicit label sets are compared."""
    a,b = parameter_labels(query),parameter_labels(candidate)
    if not a or not b:
        return 0.
    return 20.*len(a ^ b)/len(a | b)


def apply_confidence(results):
    """Use the same fit curve and ambiguity ceiling for every ranked candidate.

    Scores are evidence indicators, not calibrated correctness probabilities.
    A close competitor limits ID certainty, not the geometric fit itself.
    """
    if not results:
        return
    if any(not math.isfinite(item['structural_dist']) or item['structural_dist'] < 0
           for item in results):
        raise ValueError('Structural distances must be finite and non-negative')
    best = results[0]
    # Relative separation avoids demanding a 20-point lead from an almost
    # exact match. At least a 20% improvement removes the ambiguity cap.
    # This remains a heuristic operating rule, not a learned probability.
    if len(results) > 1:
        second = max(0., results[1]['structural_dist'])
        relative_gap = max(0., second - max(0., best['structural_dist'])) / max(second, 1e-9)
        separation = relative_gap / .20
    else:
        relative_gap, separation = 1., 1.
    ceiling = min(.98, .98 * separation)
    equivalents = best.get('geometry_equivalents', [])
    if len(equivalents) > 1:
        ceiling = min(ceiling, 1. / len(equivalents))
    for item in results:
        item['fit_score'] = .98 * math.exp(-max(0., item['structural_dist']) / 30.)
        item['confidence'] = min(item['fit_score'], ceiling)
        item['ambiguity_ceiling'] = ceiling
        item['relative_separation'] = relative_gap


def match_decision(results, query):
    if not results:
        return 'no_match', 'No suitable catalog match found. This may be a new shape or an unclear drawing.'
    best = results[0]
    # A conservative abstention boundary in the existing angular score units.
    # This is not a calibrated novelty probability and never adds a DB record.
    if query.get('foreground_coverage',1.) < .95:
        return 'review', 'Substantial disconnected strokes were excluded from the trace. Crop to one shape or provide a clearer drawing.'
    if not math.isfinite(best['structural_dist']) or best['structural_dist'] < 0:
        return 'review', 'The geometry score is invalid. Retry with a clearer image; no match is accepted.'
    if min(query.get('path_coverage',1.),best.get('signature',{}).get('path_coverage',1.)) < .95:
        return 'review', 'Some foreground branches could not be traced. Review the drawing before choosing an ID.'
    if best['structural_dist'] >= NO_MATCH_DISTANCE:
        return 'no_match', (
            f'No suitable catalog match found: geometric fit is at or below '
            f'{MIN_GEOMETRIC_FIT:.1%}. This may be a shape outside the catalog '
            'or an unclear drawing; database absence is not confirmed.')
    if len(best.get('geometry_equivalents',[])) > 1:
        return 'review', 'Multiple catalog IDs have identical extracted geometry. Dimensions or other identifying details are required.'
    if best.get('fit_score',best.get('confidence',0)) < .65:
        return 'review', 'The nearest candidate has a weak geometric fit. Check the extracted centerline, bends, arcs and terminal details.'
    if best.get('confidence',0) < .65:
        return 'review', 'The leading candidates are too close to select a unique ID reliably. Compare their dimensions and local details.'
    return 'matched', 'A compatible catalog candidate was found; verify dimensions before use.'
