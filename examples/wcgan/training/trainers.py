from typing import Union, List

from torch.utils.data import DataLoader

from kerosene.configs.configs import RunConfiguration
from kerosene.training.trainers import Trainer, ModelTrainer

GENERATOR_A_TO_B, GENERATOR_B_TO_A, DISCRIMINATOR_B, DISCRIMINATOR_A = 0, 1, 2, 3
REAL_A, REAL_B = 0, 1


class CycleGanTrainer(Trainer):

    def __init__(self, name, lambda_gan_A, lambda_gan_B, lambda_forward_cycle, lambda_backward_cycle,
                 train_data_loader: DataLoader, valid_data_loader: DataLoader,
                 test_data_loader: Union[DataLoader, None], model_trainers: Union[List[ModelTrainer], ModelTrainer],
                 run_config: RunConfiguration):
        super().__init__(name, train_data_loader, valid_data_loader, test_data_loader, model_trainers, run_config)
        self._lambda_gan_A = lambda_gan_A
        self._lambda_gan_B = lambda_gan_B
        self._lambda_forward_cycle = lambda_forward_cycle
        self._lambda_backward_cycle = lambda_backward_cycle

    def backward_D(self, D: ModelTrainer, real, fake):
        # Real
        pred_real = D(real)
        loss_D_real = D.compute_loss("Pred Real", pred_real, None)
        D.update_train_loss("Pred Real", loss_D_real)

        # Fake
        pred_fake = D(fake.detach())
        loss_D_fake = D.compute_loss("Pred Fake", pred_fake, None)
        D.update_train_loss("Pred Fake", loss_D_fake)

        loss_D = loss_D_fake - loss_D_real
        loss_D.backward()

        for p in D.model.parameters():
            p.data.clamp_(-0.01, 0.01)

    def train_step(self, inputs, target):
        real_A, real_B = inputs
        D_A, D_B, GEN_A_B, GEN_B_A = self._model_trainers[DISCRIMINATOR_A], self._model_trainers[
            DISCRIMINATOR_B], self._model_trainers[GENERATOR_A_TO_B], self._model_trainers[GENERATOR_B_TO_A]

        GEN_A_B.zero_grad()
        GEN_B_A.zero_grad()

        # GAN loss D_B(G_A-B(A))
        fake_B = GEN_A_B(real_A)
        pred_fake = D_B(fake_B)
        loss_GEN_A_B = -1 * GEN_A_B.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_A
        GEN_A_B.update_train_loss("Pred Fake", loss_GEN_A_B)

        # GAN loss D_A(G_B-A(B))
        fake_A = GEN_B_A(real_B)
        pred_fake = D_A(fake_A)
        loss_GEN_B_A = -1 * GEN_B_A.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_B
        GEN_B_A.update_train_loss("Pred Fake", loss_GEN_B_A)

        # Forward cycle loss
        rec_A = GEN_B_A(fake_B)
        forward_cycle_loss = GEN_B_A.compute_loss("Forward Cycle Loss", rec_A, real_A) * self._lambda_forward_cycle
        GEN_B_A.update_train_loss("Forward Cycle Loss", forward_cycle_loss)

        # Backward cycle loss
        rec_B = GEN_A_B(fake_A)
        backward_cycle_loss = GEN_A_B.compute_loss("Backward Cycle Loss", rec_B, real_B) * self._lambda_backward_cycle
        GEN_A_B.update_train_loss("Backward Cycle Loss", backward_cycle_loss)

        loss_G = loss_GEN_A_B + loss_GEN_B_A + forward_cycle_loss + backward_cycle_loss
        loss_G.backward()

        GEN_A_B.step()
        GEN_B_A.step()

        for iter_critic in range(5):
            D_A.zero_grad()
            self.backward_D(D_A, real_A, fake_A)
            D_A.step()

        for iter_critic in range(5):
            D_B.zero_grad()
            self.backward_D(D_B, real_B, fake_B)
            D_B.step()

    def validate_step(self, inputs, target):
        real_A, real_B = inputs
        D_A, D_B, GEN_A_B, GEN_B_A = self._model_trainers[DISCRIMINATOR_A], self._model_trainers[
            DISCRIMINATOR_B], self._model_trainers[GENERATOR_A_TO_B], self._model_trainers[GENERATOR_B_TO_A]

        # GAN loss D_B(G_A-B(A))
        fake_B = GEN_A_B(real_A)
        pred_fake = D_B(fake_B)
        loss_GEN_A_B = -1 * GEN_A_B.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_A
        GEN_A_B.update_valid_loss("Pred Fake", loss_GEN_A_B)

        # GAN loss D_A(G_B-A(B))
        fake_A = GEN_B_A(real_B)
        pred_fake = D_A(fake_A)
        loss_GEN_B_A = -1 * GEN_B_A.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_B
        GEN_B_A.update_valid_loss("Pred Fake", loss_GEN_B_A)

        # Forward cycle loss
        rec_A = GEN_B_A(fake_B)
        forward_cycle_loss = GEN_B_A.compute_loss("Forward Cycle Loss", rec_A, real_A) * self._lambda_forward_cycle
        GEN_B_A.update_valid_loss("Forward Cycle Loss", forward_cycle_loss)

        # Backward cycle loss
        rec_B = GEN_A_B(fake_A)
        backward_cycle_loss = GEN_A_B.compute_loss("Backward Cycle Loss", rec_B, real_B) * self._lambda_backward_cycle
        GEN_A_B.update_valid_loss("Backward Cycle Loss", backward_cycle_loss)

    def test_step(self, inputs, target):
        real_A, real_B = inputs
        D_A, D_B, GEN_A_B, GEN_B_A = self._model_trainers[DISCRIMINATOR_A], self._model_trainers[
            DISCRIMINATOR_B], self._model_trainers[GENERATOR_A_TO_B], self._model_trainers[GENERATOR_B_TO_A]

        # GAN loss D_B(G_A-B(A))
        fake_B = GEN_A_B(real_A)
        pred_fake = D_B(fake_B)
        loss_GEN_A_B = -1 * GEN_A_B.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_A
        GEN_A_B.update_test_loss("Pred Fake", loss_GEN_A_B)

        # GAN loss D_A(G_B-A(B))
        fake_A = GEN_B_A(real_B)
        pred_fake = D_A(fake_A)
        loss_GEN_B_A = -1 * GEN_B_A.compute_loss("Pred Fake", pred_fake, None) * self._lambda_gan_B
        GEN_B_A.update_test_loss("Pred Fake", loss_GEN_B_A)

        # Forward cycle loss
        rec_A = GEN_B_A(fake_B)
        forward_cycle_loss = GEN_B_A.compute_loss("Forward Cycle Loss", rec_A, real_A) * self._lambda_forward_cycle
        GEN_B_A.update_test_loss("Forward Cycle Loss", forward_cycle_loss)

        # Backward cycle loss
        rec_B = GEN_A_B(fake_A)
        backward_cycle_loss = GEN_A_B.compute_loss("Backward Cycle Loss", rec_B, real_B) * self._lambda_backward_cycle
        GEN_A_B.update_test_loss("Backward Cycle Loss", backward_cycle_loss)

    def scheduler_step(self):
        for model_trainer in self.model_trainers:
            model_trainer.scheduler_step()
