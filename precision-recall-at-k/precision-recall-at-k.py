def precision_recall_at_k(recommended: list, relevant: list, k: int) -> list[float]:
    """
    Returns [precision, recall] as a list of two floats.
    """
    ans=[]
    if k <= 0:
        return [0.0, 0.0] 
    top_k = recommended[:k]
    relevant_set = set(relevant) 
    hits = sum(1 for item in top_k if item in relevant_set)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return [precision, recall]   
    