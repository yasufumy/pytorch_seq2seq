import math

import torch
from torch.autograd import Variable
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def log_training_average_nll(trainer, logger):
    window_size = trainer.current_iteration
    nnl = sum(trainer.training_history[-window_size:]) / window_size
    log_str = 'Average NNL for Training Data: {:.4f}'.format(nnl)
    logger(log_str)
    trainer.current_validation_iteration = 0


def get_log_validation_ppl(val_data):
    num_validation_labels = sum(len(y) for y in val_data)

    def log_validation_ppl(trainer, logger):
        window_size = trainer.current_validation_iteration
        ppl = math.exp(sum(trainer.validation_history[-window_size:]) /
                       num_validation_labels)
        log_str = 'Perplexity for Validation Data: {:.4f}'.format(ppl)
        logger(log_str)
        # TODO: update these lines below when state is added to trainer
        if not getattr(trainer, 'state', None):
            trainer.state = {}
        trainer.state['ppl'] = ppl
    return log_validation_ppl


class TeacherForceUpdater:
    def __init__(self, model, optimizer, loss_func, gradient_clipping=None):
        self._model = model
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._gradient_clipping = gradient_clipping

    def __call__(self, batch):
        self._model.train()
        self._optimizer.zero_grad()
        xs, src_lengths = batch.src
        src_lengths = Variable(src_lengths.unsqueeze(0), requires_grad=False)
        loss = self._loss_func(xs, src_lengths, batch.trg)
        loss = torch.mean(loss)
        loss.backward()
        if self._gradient_clipping:
            torch.nn.utils.clip_grad_norm(
                self._model.parameters(), self._gradient_clipping)
        self._optimizer.step()
        return loss.data[0] * xs.size(1)


class TeacherForceInference:
    def __init__(self, model, infer_func):
        self._model = model
        self._infer_func = infer_func

    def __call__(self, batch):
        self._model.eval()
        xs, src_lengths = batch.src
        src_lengths = Variable(src_lengths.unsqueeze(0), volatile=True)
        loss = self._infer_func(xs, src_lengths, batch.trg)
        loss = torch.mean(loss)
        return loss.data[0] * xs.size(1)


class ComputeBleu:
    def __init__(self, model, refs, translater):
        self._model = model
        self._refs = [[y] for y in refs]
        self._translater = translater

    def __call__(self, trainer, test_iter, filename, logger=print):
        self._model.load_state_dict(torch.load(filename))
        self._model.eval()
        hyps = []
        for batch in test_iter:
            hyps.extend(self._translater(*batch.src))
        bleu = 100 * corpus_bleu(
            self._refs, hyps, smoothing_function=SmoothingFunction().method1)
        log_str = 'BLEU score for Test Data: {:.4f}'.format(bleu)
        logger(log_str)


class BestModelSnapshot:
    def __init__(self, model, metric, initial_value, compare_func):
        self._model = model
        self._metric = metric
        self._best_value = initial_value
        self._compare_func = compare_func

    def __call__(self, trainer, filename, logger=print):
        metric = trainer.state[self._metric]
        if self._compare_func(metric, self._best_value):
            self._best_value = metric
            logger('Saving model to {}'.format(filename))
            torch.save(self._model.state_dict(), filename)
