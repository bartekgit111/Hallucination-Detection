import numpy as np


def fit(model, data, epochs=3, print_update=10000):
    for epoch in range(epochs):

        indices = np.arange(len(data.text_as_indexes))
        np.random.shuffle(indices)

        loss_sum = 0
        step = 0

        for i in indices:

            center = data.text_as_indexes[i]

            for j in range(i - data.window_size, i + data.window_size + 1):

                if j == i or j < 0 or j >= len(data.text_as_indexes):
                    continue

                context = data.text_as_indexes[j]

                negatives = data.generate_negative_words(context)

                loss = model.train_step(center, context, negatives)

                loss_sum += loss
                step += 1

                if step % print_update == 0:
                    print(f"epoch {epoch}, step {step}, avg loss {loss_sum / print_update}")
                    loss_sum = 0

        print(f"END epoch {epoch}")
