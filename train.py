import os

import tensorflow as tf


def get_metric_value(metric):
    return metric.result().numpy().astype(float)


def customized_training_loop(train_iterator,
                             eval_iterator,
                             model,
                             loss_fn,
                             metric_fn,
                             step,
                             model_dir,
                             inner_model_dir,
                             summary_dir,
                             train_summary_interval,
                             eval_summary_interval,
                             epochs):
    # initialize things that are needed to train model
    model, inner_model = model
    optimizer = model.optimizer
    vars = model.trainable_variables

    cur_epoch = tf.Variable(1, dtype=tf.int32)

    train_logs = {}
    eval_logs = {}

    train_loss_metric = tf.keras.metrics.Mean(
        'train_loss', dtype=tf.float32)
    eval_loss_metric = tf.keras.metrics.Mean(
        'eval_loss', dtype=tf.float32)

    train_metrics = metric_fn() if metric_fn else []
    eval_metrics = [
        metric.__class__.from_config(metric.get_config())
        for metric in train_metrics
    ]

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, 'train'))
    eval_summary_writer = tf.summary.create_file_writer(
        os.path.join(summary_dir, 'eval'))

    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer,
                               cur_epoch=cur_epoch,
                               global_step=optimizer.iterations)
    inner_ckpt = tf.train.Checkpoint(inner_model=inner_model)

    # check whether there are checkpoint files for model or not.
    latest_ckpt_file = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt_file:
        ckpt.restore(latest_ckpt_file)
        print('Restoring checkpoint from %s' % latest_ckpt_file)

    global_step = optimizer.iterations.numpy()
    last_summary_step = global_step

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = loss_fn(labels, outputs)

        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        train_loss_metric.update_state(loss)
        for metric in train_metrics:
            metric.update_state(labels, outputs)

    def eval_step(inputs, labels):
        outputs = model(inputs)
        eval_loss = loss_fn(labels, outputs)

        eval_loss_metric.update_state(eval_loss)
        for metric in eval_metrics:
            metric.update_state(labels, outputs)

    # start training
    for epoch in range(cur_epoch.numpy(), epochs + 1):
        # reset metrics for train
        train_loss_metric.reset_states()
        for metric in train_metrics + model.metrics:
            metric.reset_states()

        for inputs, labels in train_iterator:
            train_step(inputs, labels)

            if global_step >= last_summary_step + train_summary_interval:
                with train_summary_writer.as_default():
                    for metric in [train_loss_metric] + train_metrics + model.metrics:
                        metric_value = get_metric_value(metric)
                        train_logs[metric.name] = metric_value
                        tf.summary.scalar(metric.name, metric_value, step=global_step)

                    print('train_result, cur_epoch: %d, cur_step: %d' % (epoch, global_step))
                    print(train_logs, end='\n\n')
                    train_logs.clear()

                last_summary_step = global_step

            global_step += step

        # start evaluation
        eval_loss_metric.reset_states()
        for metric in eval_metrics + model.metrics:
            metric.reset_states()

        for cur_eval_step, (inputs, labels) in enumerate(eval_iterator):
            eval_step(inputs, labels)

            if cur_eval_step % eval_summary_interval == 0:
                with eval_summary_writer.as_default():
                    for metric in [eval_loss_metric] + eval_metrics + model.metrics:
                        metric_value = get_metric_value(metric)
                        eval_logs[metric.name] = metric_value
                        tf.summary.scalar(metric.name, metric_value, step=global_step+cur_eval_step)

                    print('eval_result, cur_epoch: %d' % (epoch))
                    print(eval_logs, end='\n\n')
                    with open('eval_logs.txt', 'at') as f:
                        eval_output = 'cur_epoch: %d cur_step: %d ' % (epoch, cur_eval_step)
                        for metric, value in eval_logs.items():
                            eval_output += '%s: %.2f ' % (metric, value)
                        eval_output += '\n\n'

                        f.write(eval_output)

                    eval_logs.clear()

        # update cur_epoch
        cur_epoch.assign_add(1)

        # save checkpoint for ckpt, inner_ckpt
        ckpt.save(os.path.join(model_dir, '%d_step.ckpt' % (global_step)))
        inner_ckpt.save(os.path.join(inner_model_dir, '%d_step.ckpt' % (global_step)))

    print('training done')