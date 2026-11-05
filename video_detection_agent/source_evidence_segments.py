"""Lossless punctuation segments for model selection, never factual judgments.

All original characters survive in order, including trailing incomplete text.
Segments are quotations, not asserted-independent or verified atomic facts.
"""
import hashlib
import re


def segment_evidence(evidence):
    candidates, bindings = [], []
    for source_index, item in enumerate(evidence):
        text = item['description']
        if not isinstance(text, str): raise ValueError('Observation description must be text')
        cursor = 0; spans = []
        for match in re.finditer(r'[。！？]|[.!?](?=\s|$)', text):
            end = match.end()
            spans.append((cursor, end)); cursor = end
        if cursor < len(text): spans.append((cursor, len(text)))
        if not spans: spans = [(0, len(text))]
        # Attach punctuation/whitespace-only tails to the previous quote.
        merged = []
        for start, end in spans:
            if merged and not any(c.isalnum() for c in text[start:end]):
                merged[-1] = (merged[-1][0], end)
            else: merged.append((start, end))
        assert ''.join(text[a:b] for a,b in merged) == text
        for start, end in merged:
            candidates.append({'timestamp': item['timestamp'], 'description': text[start:end]})
            bindings.append({'candidate_index': len(candidates)-1, 'source_evidence_index': source_index,
                             'source_span': [start, end],
                             'source_description_sha256': hashlib.sha256(text.encode()).hexdigest(),
                             'semantic_atomicity': 'NOT_ASSERTED', 'factual_support': 'NOT_ASSERTED'})
    return candidates, bindings


def verify_manifest(pack):
    if pack.get('version') != 'source-evidence-segments-v1':
        raise ValueError('Unknown evidence segmentation protocol')
    expected, bindings = segment_evidence(pack['source_observations'])
    if pack['candidates'] != expected or pack['bindings'] != bindings:
        raise ValueError('Evidence candidates do not match exact lossless original spans')


def selected_evidence(pack, observations, index):
    """Validate provenance only, without assigning factual or policy support."""
    verify_manifest(pack)
    if observations != pack['source_observations']:
        raise ValueError('Candidate text differs from actual model observations')
    if index is None:
        return []
    if type(index) is not int or not 0 <= index < len(pack['candidates']):
        raise ValueError('Selected evidence index is not an actual candidate')
    return [pack['candidates'][index]]
