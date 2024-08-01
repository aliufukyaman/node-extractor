def similarity_score(gt_string, pred_string):
    # Convert strings to lists
    gt_list = gt_string.split(', ')
    pred_list = pred_string.split(', ')

    # Remove space characters from each string in the list
    pred_list = [s.replace(' ', '') for s in pred_list]

    # If the two lists are exactly the same, return 1 directly
    if gt_list == pred_list:
        return 1.0

    # Both lists must contain at least one element
    if not gt_list or not pred_list:
        return 0.0

    # Get the lengths of the lists
    gt_len = len(gt_list)
    pred_len = len(pred_list)

    # Find the exact matches in the correct positions
    exact_matches = sum(1 for gt, pred in zip(gt_list, pred_list) if gt == pred)

    # Find all the correct elements
    total_matches = sum(1 for gt in gt_list if gt in pred_list)

    # To find unordered matches
    partial_matches = total_matches - exact_matches

    # Assign a weight of 0.7 for correct positions and 0.3 for unordered matches
    score = (exact_matches / gt_len) * 0.7 + (partial_matches / gt_len) * 0.3

    # Restrict the score between 0 and 1
    score = max(0.0, min(1.0, score))

    return score

def calculate_similarity_score(ground_truth, test_string):
    # Split the strings into lists of words
    ground_truth_list = ground_truth.split(', ')
    test_list = test_string.split(', ')

    # Initialize the match count
    match_count = 0

    # Calculate the matches considering the order
    for gt_word, test_word in zip(ground_truth_list, test_list):
        if gt_word.replace(' ', '') == test_word.replace(' ', ''):
            match_count += 1

    # Calculate the score
    score = match_count / len(ground_truth_list)

    return round(score, 2)
