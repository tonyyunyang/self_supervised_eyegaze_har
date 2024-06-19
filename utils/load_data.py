def get_fully_supervised_pretrain_indices(subjects, leave_out_subject, last_index):
    pretrain_test_indices = []
    pretrain_train_indices = []

    # Ensure subjects are sorted by their starting indices to maintain order
    sorted_subjects = sorted(subjects.items(), key=lambda x: x[1])

    # Find the indices for the leave-out subject
    for i, (subject, start) in enumerate(sorted_subjects):
        # Calculate the end index based on whether it's the last subject in the list
        end = sorted_subjects[i + 1][1] if i + 1 < len(sorted_subjects) else last_index + 1

        if subject == leave_out_subject:
            pretrain_test_indices.extend(range(start, end))
        else:
            pretrain_train_indices.extend(range(start, end))

    return pretrain_test_indices, pretrain_train_indices
