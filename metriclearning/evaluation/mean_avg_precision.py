import numpy as np


def average_precision(relevant, retrieved):
    """
    relevant: set of relevant labels
    retrieved: list of retrieved documents sorted according to relevance
    """
    is_hit = np.array([x in relevant for x in retrieved])
    precision = np.cumsum(is_hit, dtype=float) / np.arange(1, len(retrieved) + 1)
    # Note that the average is over all relevant documents
    # and the relevant documents not retrieved get a precision score of zero
    precision *= is_hit
    return precision.sum() / is_hit.sum() if is_hit.sum() > 0 else 0


def mean_avg_precision(T_query, Y):
    assert isinstance(T_query, list) or (isinstance(T_query, np.ndarray) and len(T_query.shape) == 1)
    return np.mean([
        average_precision([t_q], retrieved_labels)
            for t_q, retrieved_labels in zip(T_query, Y)
    ])


def mean_avg_R_precision(T_query, Y, T_gallery):
    """
    Mean Average R-Precision at R (MAP@R)
    Similar to MAP but only retrive R nns, where R is different for every query.
    R = number of samples in the gallery se twith the same label as the query.

    https://cs.stackexchange.com/questions/67736/what-is-the-difference-between-r-precision-and-precision-at-k

    Args:
        T_query: (np.array) labels of the query images
        Y: (np.array or list of np.array's) labels of the retrieved images
            for every query
        T_gallery: labels of the entire gallery set. Is used only for
            calculation of the class sizes (to get R).
    """
    assert isinstance(T_gallery, list) or (isinstance(T_gallery, np.ndarray) and len(T_gallery.shape) == 1)

    # In case that the test set is the same as our query set.
    # the query image should be excluded
    remove_self = int(np.array_equal(T_query, T_gallery))

    class_size = {t: (T_gallery == t).sum() - remove_self for t in np.unique(T_gallery)}
    Y_list = [retrieved_labels[:class_size[tq]] for tq, retrieved_labels in zip(T_query, Y)]

    return mean_avg_precision(T_query, Y_list)


if __name__ == '__main__':
    # https://stackoverflow.com/a/40834813/5042151
    rel_ = [7, 7, 7, 7, 7, 7]
    retr_ = [7, 5, 7, 7, 7, 7, 2, 3, 1, 7]
    print(average_precision(rel_, retr_))

    rel_  = [1, 1, 1, 1, 1, 1]
    retr_ = [0, 1, 2, 3, 1, 1, 1, 0, 1, 1]
    print(average_precision(rel_, retr_))
    # outputs should be
    # 0.7749999999999999
    # 0.5211640211640212

    rel_  = [1, 1, 1, 1, 1, 1]
    rel_  = [1]
    retr_ = [0, 1, 1]
    print(average_precision(rel_, retr_))


    print('map=', mean_avg_precision([7, 1, 1],
                             [[7, 5, 7, 7, 7, 7, 2, 3, 1, 7],
                              [0, 1, 2, 3, 1, 1, 1, 0, 1, 1],
                              [0, 1, 1, 0, 0]
                             ])
         )


