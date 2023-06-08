import numpy as np
import torch


class CollectFn:
    
    def __call__(self, batch):
        """
        Takes in a batch of data and returns a dictionary of tensors
        for the neural network to use for training or evaluation.

        Args:
            batch: A list of tuples where each tuple contains a sample tensor
                   and a label tensor.

        Returns:
            A dictionary with the following possible keys and associated
            tensor values:
            - If the sample tensor is 2D and the label tensor is 1D, the
              return value has the key "batch_data" and the value is a
              tensor of the sample data.
            - If the sample tensor is 2D and the label tensor is 2D, the
              return value has the keys "pos", "neg", and "neg_labels" where
              "pos" and "neg" are tensors of the positive and negative
              samples respectively and "neg_labels" is a tensor of the negative
              label values.
            - If the sample tensor is 3D and the label tensor is 2D, the
              return value has the keys "batch_pairs" and "batch_labels"
              where "batch_pairs" is a tensor of the sample data and
              "batch_labels" is a tensor of the label values.
        """
        sample_shape = batch[0][0].shape
        label_shape = batch[0][1].shape
        batch_samples = np.vstack([x[0] for x in batch])
        batch_labels = np.vstack([x[1] for x in batch])

        if len(sample_shape) == 2:
            if label_shape[0] == 1:
                return {"batch_data": torch.from_numpy(batch_samples)}
            batch_labels = batch_labels.reshape(-1)
            batch_neg = batch_samples[batch_labels != 0]
            batch_pos = batch_samples[batch_labels == 0]
            return {
                "pos": torch.from_numpy(batch_pos).long(),
                "neg": torch.from_numpy(batch_neg).long(),
                "neg_labels": torch.from_numpy(batch_labels[batch_labels != 0]).long(),
            }

        batch_pairs = torch.from_numpy(batch_samples).long()
        batch_labels = torch.from_numpy(batch_labels).float()
        return {
            "batch_pairs": batch_pairs,
            "batch_labels": batch_labels,
        }


