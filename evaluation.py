# evaluation.py

def precision_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    hits = 0
    for item in recommended_at_k:
        if item in relevant_set:
            hits += 1

    return hits / k


def recall_at_k(recommended_items, relevant_items, k):
    recommended_at_k = recommended_items[:k]
    relevant_set = set(relevant_items)

    hits = 0
    for item in recommended_at_k:
        if item in relevant_set:
            hits += 1

    if len(relevant_set) == 0:
        return 0

    return hits / len(relevant_set)