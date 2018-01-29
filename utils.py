import math


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
    return log_validation_ppl


def early_stopping_handler(trainer, lookback=1):
    history = trainer.validation_history
    if len(history) <= lookback:
        return
    last_ppl = history[-1]
    if not any(x > last_ppl for x in history[-lookback-1:-1]):
        trainer.terminate()
