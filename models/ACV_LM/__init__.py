from models.ACV_LM.acv import ACVNet
from models.ACV_LM.loss import model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, model_loss_test

__models__ = {
    "acvnet": ACVNet
}
