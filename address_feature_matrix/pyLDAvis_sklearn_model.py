"""
Re-implementation of sklearn-compatibility for pyLDAvis.
"""

from pyLDAvis import prepare as vis_prepare
import numpy as np


def _extract_data(lda_model, dtm, vectorizer):
    doc_lengths = dtm.sum(axis=1).A.ravel()
    term_freqs = np.asarray(dtm.sum(axis=0)).ravel()

    vocab = vectorizer.get_feature_names_out()
    term_topic = lda_model.components_
    topic_term_dists = (term_topic.T / term_topic.sum(axis=1)).T
    doc_topic_dists = lda_model.transform(dtm)

    return topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_freqs


def prepare(lda_model, dtm, vectorizer):
    topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_freqs = _extract_data(lda_model, dtm, vectorizer)
    return vis_prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths,
        vocab=vocab,
        term_frequency=term_freqs
    )
