from torch import nn
import torch


class BasicModel(nn.Module):
    """
    BasicModel is a base model for developing custom recommendation systems.

    Methods:
        get_user_rating(users): It should return the ratings for given users.
        parameters_norm(): Returns the norm of the parameters in the model.
    """
    def __init__(self):
        """
        Initializes the BasicModel.
        """
        super(BasicModel, self).__init__()

    def get_user_rating(self, users):
        """
        抽象方法，用于获取给定用户的评分。
        子类中必须实现这个方法，或者抛出NotImplementedError异常。
        Args:
            users: The users for whom to retrieve ratings.
        """
        raise NotImplementedError

    def parameters_norm(self):
        """
       计算模型参数的范数（norm），这里仅返回一个零张量，在子类中可以按需重写。
        """
        return torch.tensor(0)
