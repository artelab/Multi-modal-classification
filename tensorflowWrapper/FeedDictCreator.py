class FeedDictCreator(object):

    @staticmethod
    def create_feed_dict(model_tensor, train_batch, train_images_batch, dropout_keep_prob):
        feed_dict = {
            model_tensor.input_x: train_batch[0],
            model_tensor.input_y: train_batch[1],
            model_tensor.input_mask: train_images_batch,
            model_tensor.dropout_keep_prob: dropout_keep_prob
        }
        return feed_dict
